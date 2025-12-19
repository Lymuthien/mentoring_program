import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from recruit_restaurant_visitor_forecasting.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_COL,
    VISITORS_NBR_COL,
    CITY_COL,
    TOTAL_RESERVES_COL,
    RES_VISITORS_DIFF_COL,
    TOTAL_RESERVES_NBR_COL,
    RES_VISITORS_DIFF_NBR_COL,
)
from recruit_restaurant_visitor_forecasting.features import (
    add_lags,
    add_basic_stats,
    add_neighbors_stats,
    add_last_month_visitors,
    add_reserves_difference,
)
from recruit_restaurant_visitor_forecasting.feature_names import (
    lag_col,
    last_month_col,
    nbrs_col,
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

    df = df.drop(columns=cols_to_drop)
    df = df.rename(columns=cols_to_rename)

    return df


def update_features_for_date(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    lags: tuple = (1, 7, 28),
) -> pd.DataFrame:
    id_col = AIR_RESTAURANT_ID_COL

    start_date = current_date - pd.DateOffset(months=1)
    df = df[
        (df[VISIT_DATE_COL] >= start_date) & (df[VISIT_DATE_COL] <= current_date)
    ].copy()
    old_df = df.copy()

    df = df.sort_values([VISIT_DATE_COL, id_col]).reset_index(drop=True)
    original_cols = set(df.columns)

    df = add_basic_stats(df, VISITORS_COL, id_col)
    df = add_neighbors_stats(df, VISITORS_COL, CITY_COL)

    df = add_last_month_visitors(df, VISITORS_COL)
    df = add_lags(df, id_col, VISITORS_COL, lags, False)
    df = _clean_merge_columns(df, original_cols)

    last_month = last_month_col(VISITORS_COL)
    lag_28 = lag_col(VISITORS_COL, 28)
    nbrs_lag_28 = lag_col(nbrs_col(VISITORS_COL), 28)

    df = add_lags(df, CITY_COL, VISITORS_NBR_COL, lags, True)
    df = _clean_merge_columns(df, original_cols)
    df[lag_28] = df[lag_28].fillna(old_df[lag_28]).fillna(df[nbrs_lag_28])
    df[last_month] = df[last_month].fillna(old_df[last_month]).fillna(df[lag_28])

    df = add_reserves_difference(
        df, VISITORS_COL, TOTAL_RESERVES_COL, RES_VISITORS_DIFF_COL
    )
    df = add_reserves_difference(
        df, VISITORS_NBR_COL, TOTAL_RESERVES_NBR_COL, RES_VISITORS_DIFF_NBR_COL
    )

    aggs = [("mean", {})]
    df = add_basic_stats(df, RES_VISITORS_DIFF_COL, id_col, aggs)
    df = add_neighbors_stats(df, RES_VISITORS_DIFF_NBR_COL, CITY_COL, aggs, False)

    df = _clean_merge_columns(df, original_cols)
    df_next_updated = df[df[VISIT_DATE_COL] == current_date].copy()

    return df_next_updated


def recursive_predict(
    model: Pipeline,
    test_features: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    drop_cols: list,
) -> tuple[pd.Series, pd.DataFrame]:
    id_col = AIR_RESTAURANT_ID_COL
    idx_cols = [id_col, VISIT_DATE_COL]
    feature_exclude = {id_col, VISIT_DATE_COL, VISITORS_COL, *drop_cols}

    combined = pd.concat(
        [train_features.copy(), test_features.copy()], ignore_index=True
    )
    combined[VISITORS_COL] = 0
    combined.loc[: len(train_features) - 1, VISITORS_COL] = train_labels.values
    combined = combined.sort_values([VISIT_DATE_COL, id_col]).reset_index(drop=True)

    test_dates = sorted(test_features[VISIT_DATE_COL].unique())
    result = pd.Series(index=test_features.index, dtype=float)

    for date in tqdm(test_dates, desc="Predicting recursively"):
        combined = combined.drop(columns=[VISITORS_NBR_COL])
        updated_features = update_features_for_date(
            combined,
            date,
        )

        missing_cols = updated_features.columns.difference(combined.columns)
        if not missing_cols.empty:
            combined[missing_cols] = np.nan

        combined = combined.set_index(idx_cols)
        updated_features = updated_features.set_index(idx_cols)
        combined.update(updated_features)
        combined = combined.reset_index()

        date_mask = (combined[VISIT_DATE_COL] == date) & (
            combined.index >= len(train_features)
        )
        current_features = combined[date_mask]
        X_current = current_features.drop(columns=feature_exclude)
        y_pred = model.predict(X_current.values)
        y_pred = np.maximum(y_pred, 0)

        combined.loc[current_features.index, VISITORS_COL] = y_pred

        test_date_mask = test_features[VISIT_DATE_COL] == date
        test_date_df = test_features[test_date_mask]
        test_date_indices = test_date_df.index
        result.loc[test_date_indices] = y_pred

    return result, combined
