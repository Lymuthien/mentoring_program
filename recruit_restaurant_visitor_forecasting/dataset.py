from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from recruit_restaurant_visitor_forecasting.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    VISIT_DATETIME_COL,
    RESERVE_DATETIME_COL,
    VISIT_DATE_COL,
)


def read_csv(relative_path: str):
    return pd.read_csv(RAW_DATA_DIR / relative_path.lstrip("/"))


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


app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
