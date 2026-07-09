import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from recruit_restaurant_visitor_forecasting.config.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_COL,
)
from recruit_restaurant_visitor_forecasting.config.features import (
    VISITORS_NBR_COL,
    CITY_COL,
    TOTAL_RES_COL,
    RES_VISITORS_DIFF_COL,
    TOTAL_RES_NBR_COL,
    RES_VISITORS_DIFF_NBR_COL,
    MEAN_PREF,
    MEDIAN_PREF,
    STD_PREF,
    MAX_PREF,
    MIN_PREF,
    VISITORS_DOW,
    VISITORS_DOW_NBRS,
    DOW_WINDOW,
    VISITORS_LAST_MONTH,
    RES_OFFSET,
)
from recruit_restaurant_visitor_forecasting.features import (
    add_lags,
    add_basic_stats,
    add_neighbors_stats,
    add_last_month_visitors,
    add_reserves_difference,
    add_dow_cum_agg,
    add_open_usually_discr_rolling,
    add_dow_rol_agg,
)
from recruit_restaurant_visitor_forecasting.config.feature_names import (
    lag_col,
    nbrs_col,
    agg_exp_col,
    agg_window_col,
)


def _clean_merge_columns(df: pd.DataFrame, original_cols: set) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = []
    cols_to_rename = {}

    x_y_pairs = {
        col[:-2]: (f"{col[:-2]}_x", col) for col in df.columns if col.endswith("_y")
    }

    for base_col, (x_col, y_col) in x_y_pairs.items():
        cols_to_drop.append(x_col)
        if base_col in original_cols:
            df[base_col] = df[y_col]
            cols_to_drop.append(y_col)
        else:
            cols_to_rename[y_col] = base_col

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    if cols_to_rename:
        df = df.rename(columns=cols_to_rename)

    return df


def update_features_for_date(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    res_visitors_diff_nbr_rolling: bool = False,
    res_visitors_diff_rolling: bool = False,
    res_visitors_diff_nbr: bool = False,
    res_visitors_diff: bool = False,
    visitors_dow_nbr: bool = False,
    update_dow_agg: bool = True,
) -> pd.DataFrame:
    id_col = AIR_RESTAURANT_ID_COL

    start_date = current_date - pd.DateOffset(months=1)
    df = df[(df[VISIT_DATE_COL] >= start_date) & (df[VISIT_DATE_COL] <= current_date)]
    old_df = df

    original_cols = set(df.columns)

    df = add_open_usually_discr_rolling(df)

    aggs = [
        (MEAN_PREF, {}),
        (MEDIAN_PREF, {}),
        (STD_PREF, {"ddof": 0}),
        (MAX_PREF, {}),
        (MIN_PREF, {}),
    ]
    df = add_basic_stats(df, VISITORS_COL, id_col, aggs)
    df = add_neighbors_stats(df, VISITORS_COL, CITY_COL)

    df = add_last_month_visitors(df, fillna=False)
    df = add_lags(df, id_col, VISITORS_COL, False)
    df = _clean_merge_columns(df, original_cols)

    lag_28 = lag_col(VISITORS_COL, 28)
    nbrs_lag_28 = lag_col(nbrs_col(VISITORS_COL), 28)

    df = add_lags(df, CITY_COL, VISITORS_NBR_COL, True)
    df[lag_28] = df[lag_28].fillna(old_df[lag_28]).fillna(df[nbrs_lag_28])
    df[VISITORS_LAST_MONTH] = (
        df[VISITORS_LAST_MONTH].fillna(old_df[VISITORS_LAST_MONTH]).fillna(df[lag_28])
    )

    aggs = [MEAN_PREF, MEDIAN_PREF, STD_PREF]
    if update_dow_agg:
        df, _ = add_dow_cum_agg(df, VISITORS_COL, VISITORS_DOW, aggs)
    if visitors_dow_nbr:
        df, _ = add_dow_cum_agg(df, VISITORS_NBR_COL, VISITORS_DOW_NBRS, aggs)
    df = add_dow_rol_agg(df, VISITORS_COL, VISITORS_DOW, aggs, DOW_WINDOW)

    if res_visitors_diff:
        df = add_reserves_difference(
            df, VISITORS_COL, TOTAL_RES_COL, RES_VISITORS_DIFF_COL
        )
    if res_visitors_diff_nbr:
        df = add_reserves_difference(
            df, VISITORS_NBR_COL, TOTAL_RES_NBR_COL, RES_VISITORS_DIFF_NBR_COL
        )

    aggs = [(MEAN_PREF, {})]
    if res_visitors_diff_rolling:
        df = add_basic_stats(df, RES_VISITORS_DIFF_COL, id_col, aggs)

    if res_visitors_diff_nbr_rolling:
        df = add_neighbors_stats(df, RES_VISITORS_DIFF_NBR_COL, CITY_COL, aggs, False)

    df_next_updated = df[df[VISIT_DATE_COL] == current_date]

    return df_next_updated


def recursive_predict(
    model: Pipeline,
    test_features: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    update_reservations: bool = True,
    update_dow_agg: bool = True,
    **kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    combined = pd.concat([train_features, test_features], ignore_index=True)
    combined = combined.copy()
    combined[VISITORS_COL] = 0.0
    combined.loc[: len(train_features) - 1, VISITORS_COL] = train_labels.values

    test_dates = sorted(test_features[VISIT_DATE_COL].unique())
    result = pd.Series(index=test_features.index, dtype=float)
    i = 1

    for date in tqdm(test_dates, desc="Predicting recursively"):
        updated_features = update_features_for_date(combined, date, update_dow_agg=update_dow_agg, **kwargs)

        missing_cols = updated_features.columns.difference(combined.columns)
        if not missing_cols.empty:
            combined[missing_cols] = np.nan

        date_mask = combined[VISIT_DATE_COL] == date

        cols_to_update = updated_features.columns.difference([VISITORS_COL])
        combined.loc[date_mask, cols_to_update] = updated_features[
            cols_to_update
        ].values

        current_features = combined[date_mask]
        cols = current_features.columns

        if update_reservations:
            res_cols = [TOTAL_RES_COL, TOTAL_RES_NBR_COL]
            agg_cols = [agg_window_col(col, MEAN_PREF, 7) for col in res_cols]
            res_mean_col = agg_window_col(TOTAL_RES_COL, MEAN_PREF, 7)
            offset = i if agg_exp_col(TOTAL_RES_COL, i) in cols else i - 1
            res_cols += agg_cols if res_mean_col in cols else []

            for col in res_cols:
                current_features[col] = current_features[agg_exp_col(col, offset)]
            current_features[RES_OFFSET] = offset

            if offset == i:
                i += 1

        X_current = current_features.drop(columns=[VISITORS_COL])
        y_pred = model.predict(X_current)
        y_pred = np.maximum(y_pred, 0)

        combined.loc[current_features.index, VISITORS_COL] = y_pred

        test_date_df = test_features[test_features[VISIT_DATE_COL] == date]
        result.loc[test_date_df.index] = y_pred

    return result, combined
