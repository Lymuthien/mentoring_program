from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from recruit_restaurant_visitor_forecasting.config import (
    PROCESSED_DATA_DIR,
    DAYS_FROM_HOL_COL,
    YEAR_COL,
    MONTH_COL,
    DAY_COL,
    YEAR_MONTH_COL,
    WEEK_COL,
    DAY_OF_WEEK_COL,
    PERCENTAGE_COL,
    DAY_STR_COL,
)

app = typer.Typer()


def get_first_str_values(s: pd.Series, n: int, sep: str = " ") -> pd.Series:
    return s.str.split(sep).str[:n].str.join(sep)


def get_opened_restaurants_pct(
    df: pd.DataFrame, visit_col: str, store_col: str
) -> pd.DataFrame:
    min_dates = df.reset_index().groupby(store_col)[visit_col].min()
    cumulative_restaurants = (
        min_dates.sort_values().value_counts().sort_index().cumsum()
    )

    daily_restaurants = df.groupby(visit_col)[store_col].nunique()

    df_pct = pd.DataFrame(
        {
            "daily_count": daily_restaurants,
            "total_existing": cumulative_restaurants.reindex(
                daily_restaurants.index, method="ffill"
            ),
        }
    )
    df_pct[PERCENTAGE_COL] = df_pct["daily_count"] / df_pct["total_existing"] * 100

    return df_pct


def add_seasonal_columns(df: pd.DataFrame):
    index: pd.DatetimeIndex = df.index

    df[YEAR_COL] = index.year
    df[MONTH_COL] = index.month
    df[DAY_COL] = index.day
    df[WEEK_COL] = index.isocalendar().week
    df[DAY_OF_WEEK_COL] = index.day_of_week
    df[DAY_STR_COL] = index.strftime("%a")
    df[YEAR_MONTH_COL] = [str(x.year) + "_" + str(x.month) for x in df.index]


def add_holiday_columns(df: pd.DataFrame, date_col: str, holiday_col: str):
    df = df.sort_values(date_col).reset_index(drop=True)

    grp_prev = df[holiday_col].cumsum()
    days_since_prev = df.groupby(grp_prev).cumcount()
    days_since_prev = days_since_prev.where(grp_prev != 0, np.iinfo(np.int64).max)

    df_rev = df.iloc[::-1].reset_index(drop=True)
    grp_next = df_rev[holiday_col].cumsum()
    days_until_next_rev = df_rev.groupby(grp_next).cumcount()
    days_until_next_rev = days_until_next_rev.where(
        grp_next.values != 0, np.iinfo(np.int64).max
    )
    days_until_next = days_until_next_rev.iloc[::-1].reset_index(drop=True)

    before_mask = days_until_next < days_since_prev
    signed = pd.Series(
        np.where(before_mask, -days_until_next, days_since_prev), index=df.index
    )

    df[DAYS_FROM_HOL_COL] = signed.astype(int)

    return df


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
