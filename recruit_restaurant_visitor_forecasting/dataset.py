import pandas as pd
from recruit_restaurant_visitor_forecasting.config.config import RESERVE_DATETIME_COL, VISIT_DATE_COL, VISIT_DATETIME_COL


def standardize_date(df: pd.DataFrame, col: str):
    df[col] = pd.to_datetime(df[col])


def prepare_datetime_columns(df: pd.DataFrame):
    standardize_date(df, VISIT_DATETIME_COL)
    standardize_date(df, RESERVE_DATETIME_COL)
    df[VISIT_DATE_COL] = df[VISIT_DATETIME_COL].dt.normalize()


def split_reserve_by_date(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, col: str = VISIT_DATETIME_COL
):
    occurred = df[df[col] < cutoff_date]
    target = df[df[col] >= cutoff_date]

    return occurred, target


def fill_missing_dates(
    df: pd.DataFrame,
    date_col: str,
    id_col: str,
    visitors_col: str,
    min_dates: pd.Series = None,
) -> pd.DataFrame:
    if min_dates is None:
        min_dates = df.groupby(id_col)[date_col].min()

    global_end = df[date_col].max()
    min_dates = min_dates[min_dates <= global_end]

    date_ranges = min_dates.apply(
        lambda first_date: pd.date_range(start=first_date, end=global_end, freq="D")
    )
    full_idx = date_ranges.reset_index().explode(date_col).reset_index(drop=True)

    merged = full_idx.merge(df, how="left", on=[id_col, date_col])
    merged[visitors_col] = merged[visitors_col].fillna(0).astype(int)

    merged = merged.sort_values([id_col, date_col]).reset_index(drop=True)
    return merged
