import numpy as np
import pandas as pd
import statsmodels.api as sm
from recruit_restaurant_visitor_forecasting.config import (
    AIR_DAILY_COL,
    HPG_DAILY_COL,
    HPG_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    ACTUAL_MEAN,
    PRED_MEAN,
    VISITORS_COL,
    AIR_RESTAURANT_ID_COL,
)
from recruit_restaurant_visitor_forecasting.features import add_sum_of_reserves


def get_dfs_daily_corr(
    air_df: pd.DataFrame, hpg_df: pd.DataFrame, exclude_dates: list[pd.Timestamp] = None
) -> tuple[float, pd.DataFrame]:
    air = add_sum_of_reserves(air_df, AIR_DAILY_COL)
    hpg = add_sum_of_reserves(hpg_df, HPG_DAILY_COL, HPG_RESTAURANT_ID_COL)
    air_daily = air.groupby(VISIT_DATE_COL)[AIR_DAILY_COL].mean()
    hpg_daily = hpg.groupby(VISIT_DATE_COL)[HPG_DAILY_COL].mean()

    combined = pd.concat([air_daily, hpg_daily], axis=1)

    if exclude_dates:
        exclude_dates = pd.to_datetime(exclude_dates)
        combined = combined[~combined.index.isin(exclude_dates)]

    corr = combined[AIR_DAILY_COL].corr(combined[HPG_DAILY_COL])

    return corr, combined


def find_reservations_exceed_visitors(
    grouping_df: pd.DataFrame,
    merging_df: pd.DataFrame,
    store_col: str,
    date_col: str,
    reserve_visitors_col,
    visitors_col,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_visitors = (
        grouping_df.groupby([store_col, date_col])[reserve_visitors_col]
        .sum()
        .reset_index()
    )
    merged_df = pd.merge(
        merging_df,
        daily_visitors,
        on=[store_col, date_col],
        how="outer",
    ).fillna(0)

    problematic_rows = merged_df[
        (merged_df[visitors_col] < merged_df[reserve_visitors_col])
    ][[date_col, store_col, visitors_col, reserve_visitors_col]]

    return problematic_rows, merged_df


def filter_by_column_comparison(
    df: pd.DataFrame, greater_column: str, smaller_column: str
) -> pd.DataFrame:
    return df[df[greater_column] > df[smaller_column]]


def compute_acf(df: pd.DataFrame, col: str, nlags: int = 7) -> pd.DataFrame:
    values = df.values if isinstance(df, pd.Series) else df[col].values

    acf_vals = sm.tsa.acf(values, nlags=nlags, fft=False, bartlett_confint=False)

    return pd.DataFrame(
        {"ACF": acf_vals}, index=pd.RangeIndex(start=0, stop=nlags + 1, name="Lag")
    )[1:]


def get_first_dates(df: pd.DataFrame, id_col: str) -> pd.Series:
    first_dates = df.groupby(id_col)[VISIT_DATE_COL].min()
    return first_dates


def get_daily_means(
    dates: pd.DataFrame, values: pd.Series | np.ndarray
) -> pd.DataFrame:
    daily_actual = (
        pd.DataFrame(
            {
                "date": dates.values,
                "actual": values.values if isinstance(values, pd.Series) else values,
            }
        )
        .groupby("date")["actual"]
        .mean()
        .reset_index()
    )
    return daily_actual


def merge_daily_pred(
    dates: pd.DataFrame, y: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    daily_actual = get_daily_means(dates, y)
    daily_actual.columns = [VISIT_DATE_COL, ACTUAL_MEAN]
    daily_pred = get_daily_means(dates, y_pred)
    daily_pred.columns = [VISIT_DATE_COL, PRED_MEAN]
    daily = daily_actual.merge(daily_pred, on=VISIT_DATE_COL)
    daily = daily.sort_values(VISIT_DATE_COL)

    return daily


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def get_format(orig_df, labels):
    labels = labels.to_frame(VISITORS_COL)
    labels["id"] = (
        orig_df[AIR_RESTAURANT_ID_COL] + "_" + orig_df[VISIT_DATE_COL].astype(str)
    )
    return labels.reset_index(drop=True)
