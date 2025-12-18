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


def _clean_merge_columns(df: pd.DataFrame, original_cols: set) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = []
    cols_to_rename = {}

    x_y_pairs = {}
    standalone_y = []
    standalone_x = []
    
    for col in df.columns:
        if col.endswith('_y'):
            base_col = col[:-2]
            x_col = f"{base_col}_x"
            if x_col in df.columns:
                x_y_pairs[base_col] = (x_col, col)
            else:
                standalone_y.append((base_col, col))
        elif col.endswith('_x'):
            base_col = col[:-2]
            y_col = f"{base_col}_y"
            if y_col not in df.columns:
                standalone_x.append((base_col, col))

    for base_col, (x_col, y_col) in x_y_pairs.items():
        cols_to_drop.append(x_col)
        if base_col in original_cols:
            df[base_col] = df[y_col]
            cols_to_drop.append(y_col)
        else:
            cols_to_rename[y_col] = base_col

    for base_col, y_col in standalone_y:
        if base_col in original_cols:
            df[base_col] = df[y_col]
            cols_to_drop.append(y_col)
        else:
            cols_to_rename[y_col] = base_col

    for base_col, x_col in standalone_x:
        if base_col not in original_cols:
            cols_to_rename[x_col] = base_col

    df = df.drop(columns=cols_to_drop)
    df = df.rename(columns=cols_to_rename)
    
    return df


def update_features_for_date(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    lags: tuple = (1, 7, 28),
) -> pd.DataFrame:
    id_col = AIR_RESTAURANT_ID_COL

    next_date = current_date + pd.DateOffset(days=1)
    df_next = df[df[VISIT_DATE_COL] == next_date].copy()
    if len(df_next) == 0:
        return df_next

    start_date = next_date - pd.DateOffset(months=1)
    df = df[
        (df[VISIT_DATE_COL] >= start_date) & (df[VISIT_DATE_COL] <= next_date)
    ].copy()

    df = df.sort_values([VISIT_DATE_COL, id_col]).reset_index(drop=True)
    original_cols = set(df.columns)

    df = add_basic_stats(df, VISITORS_COL, id_col)
    df = add_neighbors_stats(df, VISITORS_COL, CITY_COL)
    df = add_last_month_visitors(df, VISITORS_COL)

    df = add_lags(df, id_col, VISITORS_COL, lags, False)
    df = _clean_merge_columns(df, original_cols)
    df = add_lags(df, CITY_COL, VISITORS_NBR_COL, lags, True)

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
    df_next_updated = df[df[VISIT_DATE_COL] == next_date].copy()

    return df_next_updated


def recursive_predict(
    model: Pipeline,
    test_features: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    drop_cols: list,
) -> pd.Series:
    id_col = AIR_RESTAURANT_ID_COL

    combined_features = pd.concat(
        [train_features.copy(), test_features.copy()], ignore_index=True
    )
    combined_features = combined_features.drop(columns=[VISITORS_NBR_COL])
    combined_features[VISITORS_COL] = 0
    combined_features.loc[: len(train_features) - 1, VISITORS_COL] = train_labels.values
    combined_features = combined_features.sort_values(
        [VISIT_DATE_COL, id_col]
    ).reset_index(drop=True)

    test_dates = sorted(test_features[VISIT_DATE_COL].unique())
    predictions = {}

    for date in tqdm(test_dates, desc="Predicting recursively"):
        updated_features = update_features_for_date(
            combined_features.copy(),
            date,
        )

        date_mask = combined_features[VISIT_DATE_COL] == date
        date_indices = combined_features[date_mask].index

        feature_cols = [
            c
            for c in updated_features.columns
            if c not in [id_col, VISIT_DATE_COL, VISITORS_COL]
        ]

        for c in feature_cols:
            if c not in combined_features:
                combined_features[c] = np.nan

        updated_features = updated_features.set_index([id_col, VISIT_DATE_COL])

        for orig_idx in date_indices:
            store_id = combined_features.loc[orig_idx, id_col]
            feat_date = combined_features.loc[orig_idx, VISIT_DATE_COL]
            if (store_id, feat_date) in updated_features.index:
                for col in feature_cols:
                    if col in updated_features.columns:
                        combined_features.loc[orig_idx, col] = updated_features.loc[
                            (store_id, feat_date), col
                        ]

        test_mask = date_mask & (combined_features.index >= len(train_features))
        current_features = combined_features[test_mask].copy()
        X_current = current_features.drop(columns=[*drop_cols, VISITORS_COL])

        y_pred = model.predict(X_current.values)
        y_pred = np.maximum(y_pred, 0)

        for idx, pred in zip(current_features.index, y_pred):
            store_id = current_features.loc[idx, id_col]
            predictions[(store_id, date)] = pred
            combined_features.loc[idx, VISITORS_COL] = pred

    result = pd.Series(index=test_features.index, dtype=float)
    for idx in test_features.index:
        store_id = test_features.loc[idx, id_col]
        date = test_features.loc[idx, VISIT_DATE_COL]
        result.loc[idx] = predictions.get((store_id, date), 0)

    return result, combined_features


def predict_submission(
    model: Pipeline,
    submission_df: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    drop_cols: list,
) -> pd.DataFrame:
    submission_features = submission_df.copy()

    template_features = (
        train_features.groupby(AIR_RESTAURANT_ID_COL).last().reset_index()
    )

    submission_features = submission_features.merge(
        template_features.drop(columns=[VISIT_DATE_COL], errors="ignore"),
        on=AIR_RESTAURANT_ID_COL,
        how="left",
    )

    submission_features[VISIT_DATE_COL] = pd.to_datetime(
        submission_features[VISIT_DATE_COL]
    )
    if VISITORS_COL not in submission_features.columns:
        submission_features[VISITORS_COL] = 0

    predictions = recursive_predict(
        model=model,
        test_features=submission_features,
        train_features=train_features,
        train_labels=train_labels,
        drop_cols=drop_cols,
    )

    result = submission_df.copy()
    result[VISITORS_COL] = predictions.values

    return result
