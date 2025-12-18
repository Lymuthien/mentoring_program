import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from recruit_restaurant_visitor_forecasting.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_COL,
    VISITORS_NBR_COL,
    DAY_OF_WEEK_COL,
    CITY_COL,
    VISITORS_DOW_MEAN_COL,
    VISITORS_DOW_MEAN_NBR_COL,
    TOTAL_RESERVES_COL,
    RES_VISITORS_DIFF_COL,
    TOTAL_RESERVES_NBR_COL,
    RES_VISITORS_DIFF_NBR_COL,
)
from recruit_restaurant_visitor_forecasting.features import (
    add_lags,
    add_basic_stats,
    add_neighbors_stats,
    add_historical_dow_mean,
    add_last_month_visitors,
    add_reserves_difference,
    add_days_since_last_record,
)


def update_features_for_date(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    lags: tuple = (1, 7, 28),
) -> pd.DataFrame:
    id_col = AIR_RESTAURANT_ID_COL
    df = df.copy()

    df = df[df[VISIT_DATE_COL] <= current_date].copy()

    df = df.sort_values([VISIT_DATE_COL, id_col]).reset_index(drop=True)
    df = add_days_since_last_record(df, id_col, VISIT_DATE_COL)
    # add_time_based_target_encoding?

    df = add_basic_stats(df, VISITORS_COL, id_col)
    df = add_neighbors_stats(df, VISITORS_COL, CITY_COL)
    # df = update_neighbor_lags(df, CITY_COL, VISITORS_NBR_COL, lags)
    df = add_last_month_visitors(df, VISITORS_COL)

    df = add_lags(df, id_col, VISITORS_COL, lags, False)
    df = add_lags(df, CITY_COL, VISITORS_NBR_COL, lags, True)

    df = add_historical_dow_mean(df, VISITORS_COL, VISITORS_DOW_MEAN_COL)
    # df = update_neighbor_dow_mean(df, CITY_COL, VISITORS_NBR_COL, "visitors_dow_mean_nbrs")
    df = add_historical_dow_mean(
        df,
        VISITORS_NBR_COL,
        VISITORS_DOW_MEAN_NBR_COL,
    )

    df = add_reserves_difference(
        df, VISITORS_COL, TOTAL_RESERVES_COL, RES_VISITORS_DIFF_COL
    )
    df = add_reserves_difference(
        df, VISITORS_NBR_COL, TOTAL_RESERVES_NBR_COL, RES_VISITORS_DIFF_NBR_COL
    )

    aggs = [("mean", {})]
    df = add_basic_stats(df, RES_VISITORS_DIFF_COL, id_col, aggs)
    df = add_neighbors_stats(df, RES_VISITORS_DIFF_NBR_COL, CITY_COL, aggs, False)

    return df


def recursive_predict(
    model: Pipeline,
    test_features: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    drop_cols: list = None,
) -> pd.Series:
    id_col = AIR_RESTAURANT_ID_COL
    if drop_cols is None:
        drop_cols = [id_col, VISIT_DATE_COL, DAY_OF_WEEK_COL, VISITORS_NBR_COL]

    combined_features = pd.concat(
        [train_features.copy(), test_features.copy()], ignore_index=True
    )

    combined_features[VISITORS_COL] = 0
    combined_features.loc[: len(train_features) - 1, VISITORS_COL] = train_labels.values
    combined_features = combined_features.sort_values(
        [VISIT_DATE_COL, id_col]
    ).reset_index(drop=True)

    test_dates = sorted(test_features[VISIT_DATE_COL].unique())
    predictions = {}

    for date in tqdm(test_dates, desc="Predicting recursively"):
        date_mask = combined_features[VISIT_DATE_COL] == date
        test_mask = date_mask & (combined_features.index >= len(train_features))

        if not test_mask.any():
            continue

        current_features = combined_features[test_mask].copy()
        X_current = current_features.drop(columns=[*drop_cols, VISITORS_COL])

        y_pred = model.predict(X_current.values)
        y_pred = np.maximum(y_pred, 0)

        for idx, pred in zip(current_features.index, y_pred):
            store_id = current_features.loc[idx, id_col]
            predictions[(store_id, date)] = pred
            combined_features.loc[idx, VISITORS_COL] = pred

        mask_up_to_date = combined_features[VISIT_DATE_COL] <= date
        updated_features = update_features_for_date(
            combined_features[mask_up_to_date].copy(),
            date,
        )

        original_indices = combined_features[mask_up_to_date].index

        feature_cols = [
            c
            for c in updated_features.columns
            if c not in [id_col, VISIT_DATE_COL] and c in combined_features.columns
        ]

        updated_features = updated_features.set_index([id_col, VISIT_DATE_COL])

        for col in feature_cols:
            for orig_idx in original_indices:
                store_id = combined_features.loc[orig_idx, id_col]
                feat_date = combined_features.loc[orig_idx, VISIT_DATE_COL]
                if (store_id, feat_date) in updated_features.index:
                    combined_features.loc[orig_idx, col] = updated_features.loc[
                        (store_id, feat_date), col
                    ]

    result = pd.Series(index=test_features.index, dtype=float)
    for idx in test_features.index:
        store_id = test_features.loc[idx, id_col]
        date = test_features.loc[idx, VISIT_DATE_COL]
        result.loc[idx] = predictions.get((store_id, date), 0)

    return result


def predict_submission(
    model: Pipeline,
    submission_df: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
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
    )

    result = submission_df.copy()
    result[VISITORS_COL] = predictions.values

    return result
