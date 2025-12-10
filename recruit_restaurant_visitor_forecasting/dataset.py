from enum import Enum

import pandas as pd
from recruit_restaurant_visitor_forecasting.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RESERVE_DATETIME_COL,
    VISIT_DATE_COL,
    VISIT_DATETIME_COL,
)


class DataDir(str, Enum):
    RAW = "raw"
    INTERIM = "interim"
    PROCESSED = "processed"


DATA_DIRS = {
    DataDir.RAW: RAW_DATA_DIR,
    DataDir.INTERIM: INTERIM_DATA_DIR,
    DataDir.PROCESSED: PROCESSED_DATA_DIR,
}


def read_csv(relative_path: str, directory: DataDir = DataDir.RAW) -> pd.DataFrame:
    """
    :param relative_path: path to the file relative to the selected directory
    :param directory: DataDir.RAW | DataDir.INTERIM | DataDir.PROCESSED
    :return: DataFrame
    """
    base_dir = DATA_DIRS.get(directory)
    if base_dir is None:
        raise ValueError(f"Invalid directory: {directory}")
    return pd.read_csv(base_dir / relative_path.lstrip("/"))


def save_csv(
    df: pd.DataFrame, relative_path: str, directory: DataDir = DataDir.INTERIM
) -> None:
    """
    :param df: DataFrame to save
    :param relative_path: path to the file relative to the selected directory
    :param directory: DataDir.RAW | DataDir.INTERIM | DataDir.PROCESSED
    """
    base_dir = DATA_DIRS.get(directory)
    if base_dir is None:
        raise ValueError(f"Invalid directory: {directory}")
    df.to_csv(base_dir / relative_path.lstrip("/"), index=False)


def standardize_date(df: pd.DataFrame, col: str):
    df[col] = pd.to_datetime(df[col])


def prepare_datetime_columns(df: pd.DataFrame):
    standardize_date(df, VISIT_DATETIME_COL)
    standardize_date(df, RESERVE_DATETIME_COL)
    df[VISIT_DATE_COL] = df[VISIT_DATETIME_COL].dt.normalize()


def split_reserve_by_date(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, col: str = VISIT_DATETIME_COL
):
    occurred = df[df[col] < cutoff_date].copy()
    target = df[df[col] >= cutoff_date].copy()

    return occurred, target


def fill_missing_dates(
    df: pd.DataFrame, date_col: str, id_col: str, visitors_col: str, min_dates: pd.Series = None
) -> pd.DataFrame:
    df = df.copy()
    if min_dates is None:
        min_dates = df.groupby(id_col)[date_col].min()

    global_end = df[date_col].max()
    min_dates = min_dates[min_dates <= global_end]
    
    date_ranges = min_dates.apply(
        lambda first_date: pd.date_range(start=first_date, end=global_end, freq="D")
    )
    full_idx = (
        date_ranges.reset_index()
        .explode(date_col)
        .reset_index(drop=True)
    )

    merged = full_idx.merge(df, how="left", on=[id_col, date_col])
    merged[visitors_col] = merged[visitors_col].fillna(0).astype(int)

    merged = merged.sort_values([id_col, date_col]).reset_index(drop=True)
    return merged

