import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tqdm.notebook import tqdm

from recruit_restaurant_visitor_forecasting.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_COL,
    VISITORS_NBR_COL,
    DAY_OF_WEEK_COL,
    CITY_COL,
)
from recruit_restaurant_visitor_forecasting.features import (
    add_lags,
    add_basic_stats,
    add_neighbors_stats,
    add_historical_dow_mean,
    add_last_month_visitors,
    add_reserves_difference,
)


def update_visitors_for_date(
    df: pd.DataFrame,
    date: pd.Timestamp,
    predictions: pd.Series,
    id_col: str = AIR_RESTAURANT_ID_COL,
    visitors_col: str = VISITORS_COL,
) -> pd.DataFrame:
    df = df.copy()
    mask = df[VISIT_DATE_COL] == date
    
    if isinstance(predictions, pd.Series):
        # If predictions are indexed by restaurant_id
        if predictions.index.nlevels == 1:
            df.loc[mask, visitors_col] = df.loc[mask, id_col].map(predictions).fillna(0)
        # If predictions are indexed by (restaurant_id, date)
        elif predictions.index.nlevels == 2:
            for idx in df[mask].index:
                store_id = df.loc[idx, id_col]
                df.loc[idx, visitors_col] = predictions.get((store_id, date), 0)
    else:
        # If predictions is a dict mapping (store_id, date) -> value
        for idx in df[mask].index:
            store_id = df.loc[idx, id_col]
            df.loc[idx, visitors_col] = predictions.get((store_id, date), 0)
    
    return df


# def update_neighbor_lags(
#     df: pd.DataFrame,
#     grouping_col: str = CITY_COL,
#     visitors_nbr_col: str = VISITORS_NBR_COL,
#     lags: tuple = (1, 7, 28),
# ) -> pd.DataFrame:
#     """
#     Note: visitors_nbr_col should already be calculated (mean per date and grouping_col).
#     """
#     # For neighbor lags, we need to group by grouping_col and VISIT_DATE_COL first
#     # Then calculate lags grouped by grouping_col
#     df = df.copy()
#     df = df.sort_values([grouping_col, VISIT_DATE_COL])
    
#     # Group by grouping_col and date, take mean of visitors_nbr_col (should be same for all in group)
#     df_gr = df.groupby([grouping_col, VISIT_DATE_COL])[visitors_nbr_col].first().reset_index()
#     grouped = df_gr.groupby(grouping_col)[visitors_nbr_col]
    
#     new_cols = {f"{visitors_nbr_col}_lag_{lag}": grouped.shift(lag) for lag in lags}
#     df_gr = df_gr.assign(**new_cols)
#     df_gr = df_gr.drop(visitors_nbr_col, axis=1)
    
#     # Merge back
#     df = df.merge(df_gr, on=[grouping_col, VISIT_DATE_COL], how="left")
    
#     return df


# def update_neighbor_dow_mean(
#     df: pd.DataFrame,
#     grouping_col: str = CITY_COL,
#     visitors_nbr_col: str = VISITORS_NBR_COL,
#     feature_name: str = "visitors_dow_mean_nbrs",
# ) -> pd.DataFrame:
#     df = df.copy()
#     df_sorted = df.sort_values([grouping_col, VISIT_DATE_COL])
#     grouping_keys = [grouping_col, DAY_OF_WEEK_COL]

#     def shift_and_expanding_mean(series):
#         shifted = series.shift(1)
#         return shifted.expanding(min_periods=1).mean()

#     # Group by grouping_col and day_of_week, calculate expanding mean
#     df_sorted[feature_name] = df_sorted.groupby(grouping_keys)[visitors_nbr_col].transform(
#         shift_and_expanding_mean
#     )

#     return df_sorted.sort_index()


def update_features_for_date(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    lags: tuple = (1, 7, 28),
    windows: list = [7, 14, 28],
) -> pd.DataFrame:
    id_col = AIR_RESTAURANT_ID_COL
    df = df.copy()
    
    df = df[df[VISIT_DATE_COL] <= current_date].copy()
    
    df = df.sort_values([VISIT_DATE_COL, id_col]).reset_index(drop=True)
    
    df = add_lags(df, id_col, VISITORS_COL, lags, use_nbrs=False)
    df = add_basic_stats(df, VISITORS_COL, id_col, windows=windows)
    
    df = add_neighbors_stats(df, VISITORS_COL, CITY_COL, windows=windows, rename_col=True)
    # df = update_neighbor_lags(df, CITY_COL, VISITORS_NBR_COL, lags)
    df = add_lags(
        df,
        CITY_COL,
        VISITORS_NBR_COL,
        lags,
        True
    )
    df = add_historical_dow_mean(df, VISITORS_COL, "visitors_dow_mean")
    # df = update_neighbor_dow_mean(df, CITY_COL, VISITORS_NBR_COL, "visitors_dow_mean_nbrs")
    df = add_historical_dow_mean(
        df,
        VISITORS_NBR_COL,
        "visitors_dow_mean_nbrs",
    )
    
    df = add_last_month_visitors(df, VISITORS_COL)
    
    if "total_reserves" in df.columns:
        df = add_reserves_difference(df, VISITORS_COL, "total_reserves", "res_visitors_diff")
    if "total_reserves_nbrs" in df.columns:
        df = add_reserves_difference(df, VISITORS_NBR_COL, "total_reserves_nbrs", "nbr_res_visitors_diff")
    
    return df


def recursive_predict(
    model: Pipeline,
    test_features: pd.DataFrame,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    drop_cols: list = None,
    use_actuals: bool = False,
    actual_labels: pd.Series = None,
) -> pd.Series:
    id_col = AIR_RESTAURANT_ID_COL
    if drop_cols is None:
        drop_cols = [id_col, VISIT_DATE_COL, DAY_OF_WEEK_COL, VISITORS_NBR_COL]
    
    combined_features = pd.concat([
        train_features.copy(),
        test_features.copy()
    ], ignore_index=True)
    
    combined_features[VISITORS_COL] = 0
    combined_features.loc[:len(train_features)-1, VISITORS_COL] = train_labels.values
    combined_features = combined_features.sort_values([VISIT_DATE_COL, id_col]).reset_index(drop=True)
    
    test_dates = sorted(test_features[VISIT_DATE_COL].unique())
    predictions = {}
    
    for date in tqdm(test_dates, desc="Predicting recursively"):
        date_mask = combined_features[VISIT_DATE_COL] == date
        test_mask = date_mask & (combined_features.index >= len(train_features))
        
        if not test_mask.any():
            continue
        
        current_features = combined_features[test_mask].copy()
        X_current = current_features.drop(columns=drop_cols)
        
        y_pred = model.predict(X_current.values)
        y_pred = np.maximum(y_pred, 0)  
        
        for idx, pred in zip(current_features.index, y_pred):
            store_id = current_features.loc[idx, id_col]
            predictions[(store_id, date)] = pred
            combined_features.loc[idx, VISITORS_COL] = pred
        
        # if use_actuals and actual_labels is not None:
        #     for idx in current_features.index:
        #         store_id = current_features.loc[idx, id_col]
        #         actual_idx = actual_labels.index[
        #             (actual_labels.index.get_level_values(0) == store_id) & 
        #             (actual_labels.index.get_level_values(1) == date)
        #         ]
        #         if len(actual_idx) > 0:
        #             combined_features.loc[idx, VISITORS_COL] = actual_labels.loc[actual_idx[0]]
        #         else:
        #             combined_features.loc[idx, VISITORS_COL] = predictions[(store_id, date)]
        # else:
        #     for idx in current_features.index:
        #         store_id = current_features.loc[idx, id_col]
        #         combined_features.loc[idx, VISITORS_COL] = predictions[(store_id, date)]
        
        mask_up_to_date = combined_features[VISIT_DATE_COL] <= date
        updated_features = update_features_for_date(
            combined_features[mask_up_to_date].copy(),
            date,
            id_col,
            VISITORS_COL,
            CITY_COL,
            VISITORS_NBR_COL,
        )
        
        original_indices = combined_features[mask_up_to_date].index
        
        feature_cols = [c for c in updated_features.columns 
                       if c not in [id_col, VISIT_DATE_COL] and c in combined_features.columns]
        
        updated_features = updated_features.set_index([id_col, VISIT_DATE_COL])
        
        for col in feature_cols:
            for orig_idx in original_indices:
                store_id = combined_features.loc[orig_idx, id_col]
                feat_date = combined_features.loc[orig_idx, VISIT_DATE_COL]
                if (store_id, feat_date) in updated_features.index:
                    combined_features.loc[orig_idx, col] = updated_features.loc[(store_id, feat_date), col]
    
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
    
    template_features = train_features.groupby(AIR_RESTAURANT_ID_COL).last().reset_index()
    
    submission_features = submission_features.merge(
        template_features.drop(columns=[VISIT_DATE_COL], errors='ignore'),
        on=AIR_RESTAURANT_ID_COL,
        how='left'
    )
    
    submission_features[VISIT_DATE_COL] = pd.to_datetime(submission_features[VISIT_DATE_COL])
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
