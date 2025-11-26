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
    GOLDEN_WEEK_FLG,
    HOLIDAY_COL,
    OPENED_RECENTLY_FLG,
    OPEN_DATE_COL,
    IS_OPENED,
    DAYS_SINCE_LAST_RECORD,
    AIR_RESTAURANT_ID_COL,
    HPG_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    RESERVE_VISITORS_COL,
    CITY_REGION_COL,
    RESERVE_AIR,
    RESERVE_HPG,
    RESERVE_AIR_NEIGHBORS,
    TOTAL_RESERVES,
    RESERVE_HPG_NEIGHBORS,
    TOTAL_RESERVES_NEIGHBORS,
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


def add_holiday_columns(df: pd.DataFrame, date_col: str):
    df = df.sort_values(date_col).reset_index(drop=True)

    grp_prev = df[HOLIDAY_COL].cumsum()
    days_since_prev = df.groupby(grp_prev).cumcount()
    days_since_prev = days_since_prev.where(grp_prev != 0, np.iinfo(np.int64).max)

    df_rev = df.iloc[::-1].reset_index(drop=True)
    grp_next = df_rev[HOLIDAY_COL].cumsum()
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


def add_golden_week_flg(df: pd.DataFrame, year: int, date_col: str) -> pd.DataFrame:
    df = df.copy()

    start_date = pd.Timestamp(year=year, month=4, day=29)
    end_date = pd.Timestamp(year=year, month=5, day=5)

    df.loc[df[date_col].between(start_date, end_date), GOLDEN_WEEK_FLG] = 1

    return df


def add_opened_recently_flg(
    df: pd.DataFrame,
    open_dates: pd.Series,
    date_col: str,
    id_col: str,
    n_month: int = 6,
) -> pd.DataFrame:

    df = df.copy()
    df = df.merge(open_dates, left_on=id_col, right_index=True, how="left")

    threshold = df[date_col] - pd.DateOffset(months=n_month)

    df[OPENED_RECENTLY_FLG] = (df[OPEN_DATE_COL] >= threshold).astype(int)

    return df


def add_open_flg(df: pd.DataFrame, visitors_col: str):
    df = df.copy()
    df[IS_OPENED] = (df[visitors_col] > 0).astype(int)

    return df


def add_days_since_last_record(
    df: pd.DataFrame, id_col: str, date_col: str
) -> pd.DataFrame:
    df = df.copy()
    df.sort_values([id_col, date_col], inplace=True)

    is_open = df[IS_OPENED]
    prev_is_open = is_open.groupby(df[id_col]).shift(1).fillna(0)
    seg = prev_is_open.groupby(df[id_col]).cumsum()
    position = df.groupby([df[id_col], seg]).cumcount().astype(int)

    first_is_open = is_open.groupby([df[id_col], seg]).transform("first").astype(bool)
    base = np.where(first_is_open, 0, 1)

    df[DAYS_SINCE_LAST_RECORD] = base + position

    return df.reset_index(drop=True)


import pandas as pd


def add_time_based_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category_col: str,
    date_col: str,
    target_col: str,
    feature_name: str,
    min_samples: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    stats_df = train_df[train_df[target_col] > 0].copy()

    daily = stats_df.groupby(date_col)[target_col].agg(["sum", "count"]).reset_index()
    daily["g_cum_sum"] = daily["sum"].cumsum() - daily["sum"]
    daily["g_cum_cnt"] = daily["count"].cumsum() - daily["count"]
    daily["global_mean"] = daily["g_cum_sum"] / daily["g_cum_cnt"].replace(0, np.nan)

    agg = (
        stats_df.groupby([category_col, date_col])[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    agg["cum_sum"] = agg.groupby(category_col)["sum"].cumsum() - agg["sum"]
    agg["cum_cnt"] = agg.groupby(category_col)["count"].cumsum() - agg["count"]

    agg = agg.merge(
        daily[[date_col, "global_mean"]],
        on=date_col,
        how="left",
    )
    category_mean = agg["cum_sum"] / agg["cum_cnt"].replace(0, np.nan)
    smoothing = 1 / (1 + np.exp(-(agg["cum_cnt"] - min_samples)))

    agg[feature_name] = (
        agg["global_mean"] * (1 - smoothing)
        + category_mean.fillna(agg["global_mean"]) * smoothing
    )

    enc = (
        agg[[category_col, date_col, feature_name]]
        .sort_values([category_col, date_col])
        .reset_index(drop=True)
    )

    last_genre_values = enc.loc[
        enc.groupby(category_col)[date_col].idxmax(), [category_col, feature_name]
    ].reset_index(drop=True)
    last_global_mean = agg.loc[
        agg[date_col] == agg[date_col].max(), "global_mean"
    ].max()

    train_encoded = train_df.merge(enc, on=[category_col, date_col], how="left")

    test_encoded = test_df.merge(
        last_genre_values, on=[category_col], how="left"
    ).fillna(last_global_mean)

    return train_encoded, test_encoded


def add_sum_of_reserves(df: pd.DataFrame, output_col: str, id_col: str) -> pd.DataFrame:
    df = df.copy()
    df = (
        df.groupby([id_col, VISIT_DATE_COL])[RESERVE_VISITORS_COL]
        .sum()
        .rename(output_col)
    )
    return df.reset_index()


def add_neighbors_reserves(
    df: pd.DataFrame,
    reserves_col: str,
    region_col: str,
    neighbor_col: str,
    exclude_self: bool = True,
    fill_na=None,
) -> pd.DataFrame:
    df = df.copy()

    grp_sum = df.groupby([region_col, VISIT_DATE_COL])[reserves_col].transform("sum")
    grp_count = df.groupby([region_col, VISIT_DATE_COL])[reserves_col].transform(
        "count"
    )

    if exclude_self:
        neigh_sum = grp_sum - df[reserves_col]
        neigh_count = grp_count - 1
        neigh_mean = neigh_sum / neigh_count
        neigh_mean = neigh_mean.where(neigh_count > 0, np.nan)
    else:
        neigh_mean = grp_sum / grp_count

    if fill_na is not None:
        neigh_mean = neigh_mean.fillna(fill_na)

    df[neighbor_col] = neigh_mean
    return df

def add_total_reserves(df_visit: pd.DataFrame, air_res: pd.DataFrame, hpg_res: pd.DataFrame, region_col: str) -> pd.DataFrame:
    df = df_visit.copy()
    df = df.merge(
        air_res,
        on=[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL, region_col],
        how="left"
    ).merge(
        hpg_res,
        on=[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL, region_col],
        how="left"
    ).drop(HPG_RESTAURANT_ID_COL, axis=1)
    cols = [RESERVE_AIR, RESERVE_HPG, RESERVE_AIR_NEIGHBORS]
    df[cols] = df[cols].fillna(0)
    df[TOTAL_RESERVES] = df[RESERVE_AIR] + df[RESERVE_HPG]
    df.drop([RESERVE_AIR, RESERVE_HPG], axis=1, inplace=True)

    return df

def add_total_neigh_reserves(df_visit: pd.DataFrame, hpg_res: pd.DataFrame, region_col: str) -> pd.DataFrame:
    df = df_visit.copy()
    df = df.merge(
        hpg_res.groupby([VISIT_DATE_COL, region_col])[RESERVE_HPG_NEIGHBORS].median().rename('temp'),
        left_on=[VISIT_DATE_COL, region_col],
        right_index=True,
        how="left"
    )
    mask = df[RESERVE_HPG_NEIGHBORS].isna()
    df.loc[mask, RESERVE_HPG_NEIGHBORS] = df.loc[mask, 'temp']
    df.drop('temp', axis=1, inplace=True)
    df[TOTAL_RESERVES_NEIGHBORS] = df[RESERVE_HPG_NEIGHBORS] + df[RESERVE_AIR_NEIGHBORS]
    df.drop([RESERVE_HPG_NEIGHBORS, RESERVE_AIR_NEIGHBORS], axis=1, inplace=True)

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
