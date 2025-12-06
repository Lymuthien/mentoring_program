import numpy as np
import pandas as pd

from recruit_restaurant_visitor_forecasting.config import (
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
    DAYS_FROM_LAST_VISIT_COL,
    AIR_RESTAURANT_ID_COL,
    HPG_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    RESERVE_VISITORS_COL,
    RESERVE_AIR_COL,
    RESERVE_HPG_COL,
    RESERVE_AIR_NBR_COL,
    TOTAL_RESERVES_COL,
    RESERVE_HPG_NBR_COL,
    TOTAL_RESERVES_NBR_COL,
    DAYS_OF_WEEK,
    OPEN_USUALLY_COL,
    CITY_COL,
    VISITORS_COL,
)


def get_first_str_values(s: pd.Series, n: int, sep: str = " ") -> pd.Series:
    return s.str.split(sep).str[:n].str.join(sep)


def get_opened_restaurants_pct(
    df: pd.DataFrame, visit_col: str, id_col: str
) -> pd.DataFrame:
    min_dates = df.reset_index().groupby(id_col)[visit_col].min()
    cumulative_restaurants = (
        min_dates.sort_values().value_counts().sort_index().cumsum()
    )

    daily_restaurants = df.groupby(visit_col)[id_col].nunique()

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


def get_open_by_weekday_pct(
    df: pd.DataFrame, visit_col: str, visitors_col: str
) -> pd.DataFrame:
    df = df.copy()

    grouped = df.groupby([AIR_RESTAURANT_ID_COL, df[visit_col].dt.dayofweek])
    count = grouped.size()
    nonzero_count = grouped[visitors_col].agg(lambda s: s.ne(0).sum())

    pct_df = (nonzero_count / count).unstack(fill_value=0)
    cols_map = {i: day for i, day in enumerate(DAYS_OF_WEEK)}
    pct_df = pct_df.rename(columns=cols_map).reset_index()

    return pct_df


def get_open_status(pct_df: pd.DataFrame, threshold_ratio: float = 0.5):
    pct_df = pct_df.copy()
    max_pct = pct_df[DAYS_OF_WEEK].max(axis=1)
    threshold = threshold_ratio * max_pct

    open_cols = {f"{day}_open": (pct_df[day] >= threshold).astype(int) for day in DAYS_OF_WEEK}
    pct_df = pct_df.assign(**open_cols)
    pct_df = pct_df.drop(DAYS_OF_WEEK, axis=1)

    return pct_df


def add_open_usually(df: pd.DataFrame, drop_days: bool = True):
    df = df.copy()

    def get_open(row):
        day = row[DAY_OF_WEEK_COL]
        col_name = f"{day}_open"
        return row[col_name]

    df[OPEN_USUALLY_COL] = df.apply(get_open, axis=1)

    if drop_days:
        open_flags = [day + "_open" for day in DAYS_OF_WEEK]
        df = df.drop(open_flags, axis=1)
    return df


def add_seasonal_columns(df: pd.DataFrame):
    index: pd.DatetimeIndex = df.index

    df[YEAR_COL] = index.year
    df[MONTH_COL] = index.month
    df[DAY_COL] = index.day
    df[WEEK_COL] = index.isocalendar().week
    df[DAY_OF_WEEK_COL] = index.day_of_week
    df[DAY_STR_COL] = index.strftime("%a")
    df[YEAR_MONTH_COL] = index.year.astype(str) + "_" + index.month.astype(str)


def add_holiday_columns(df: pd.DataFrame, date_col: str):
    df = df.sort_values(date_col).reset_index(drop=True)

    inf = np.iinfo(np.int64).max
    grp_prev = df[HOLIDAY_COL].cumsum()
    days_since_prev = df.groupby(grp_prev).cumcount()
    days_since_prev = days_since_prev.where(grp_prev != 0, inf)

    df_rev = df.iloc[::-1].reset_index(drop=True)
    grp_next = df_rev[HOLIDAY_COL].cumsum()
    days_until_next_rev = df_rev.groupby(grp_next).cumcount()
    days_until_next_rev = days_until_next_rev.where(grp_next.values != 0, inf)
    days_until_next = days_until_next_rev.iloc[::-1].reset_index(drop=True)

    before_mask = days_until_next < days_since_prev
    signed = pd.Series(
        np.where(before_mask, -days_until_next, days_since_prev), index=df.index
    )

    df[DAYS_FROM_HOL_COL] = signed.astype(int)

    return df


def add_golden_week_flg(df: pd.DataFrame, years: list, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dates = df[date_col].values
    mask = np.zeros(len(df), dtype=bool)
        
    for year in years:
        start_date = pd.Timestamp(year=year, month=4, day=29)
        end_date = pd.Timestamp(year=year, month=5, day=5)
        mask |= (dates >= start_date) & (dates <= end_date)
        
    df[GOLDEN_WEEK_FLG] = mask.astype(int)

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


def add_days_since_last_record(
    df: pd.DataFrame, id_col: str, date_col: str
) -> pd.DataFrame:
    df = df.copy()
    df.sort_values([id_col, date_col], inplace=True)

    is_open = df[OPEN_USUALLY_COL]
    prev_is_open = is_open.groupby(df[id_col]).shift(1).fillna(0)
    seg = prev_is_open.groupby(df[id_col]).cumsum()
    position = df.groupby([df[id_col], seg]).cumcount().astype(int)

    first_is_open = is_open.groupby([df[id_col], seg]).transform("first").astype(bool)
    base = np.where(first_is_open, 0, 1)

    df[DAYS_FROM_LAST_VISIT_COL] = base + position

    return df.reset_index(drop=True)


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

    mask = (train_df[OPEN_USUALLY_COL] == 1) | (train_df[target_col] > 0)
    stats_df = train_df[mask].copy()

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
    last_global_mean = daily["global_mean"].iloc[-1] if len(daily) > 0 else 0

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


def add_nbrs_reserves(
    df: pd.DataFrame,
    reserves_col: str,
    region_col: str,
    nbr_col: str,
    exclude_self: bool = False,
    fill_na=None,
) -> pd.DataFrame:
    df = df.copy()

    grouped = df.groupby([region_col, VISIT_DATE_COL])[reserves_col]
    grp_sum = grouped.transform("sum")
    grp_count = grouped.transform("count")

    if exclude_self:
        nbr_count = grp_count - 1
        nbr_mean = (grp_sum - df[reserves_col]) / nbr_count
        nbr_mean = nbr_mean.where(nbr_count > 0, np.nan)
    else:
        nbr_mean = grp_sum / grp_count

    if fill_na is not None:
        nbr_mean = nbr_mean.fillna(fill_na)

    df[nbr_col] = nbr_mean
    return df


def add_total_reserves(
    df_visit: pd.DataFrame,
    air_res: pd.DataFrame,
    hpg_res: pd.DataFrame,
    region_col: str,
) -> pd.DataFrame:
    df = df_visit.copy()
    merge_columns = [AIR_RESTAURANT_ID_COL, VISIT_DATE_COL, region_col]
    df = (
        df.merge(air_res, on=merge_columns, how="left")
        .merge(hpg_res, on=merge_columns, how="left")
        .drop(HPG_RESTAURANT_ID_COL, axis=1)
    )
    cols = [RESERVE_AIR_COL, RESERVE_HPG_COL, RESERVE_AIR_NBR_COL]
    df[cols] = df[cols].fillna(0)
    df[TOTAL_RESERVES_COL] = df[RESERVE_AIR_COL] + df[RESERVE_HPG_COL]
    df.drop([RESERVE_AIR_COL, RESERVE_HPG_COL], axis=1, inplace=True)

    return df


def add_total_nbr_reserves(
    df_visit: pd.DataFrame, hpg_res: pd.DataFrame, region_col: str
) -> pd.DataFrame:
    df = df_visit.copy()
    df = df.merge(
        hpg_res.groupby([VISIT_DATE_COL, region_col])[RESERVE_HPG_NBR_COL]
        .median()
        .rename("temp"),
        left_on=[VISIT_DATE_COL, region_col],
        right_index=True,
        how="left",
    )

    df[RESERVE_HPG_NBR_COL] = df[RESERVE_HPG_NBR_COL].fillna(df["temp"])
    df = df.drop("temp", axis=1)
    df[TOTAL_RESERVES_NBR_COL] = df[RESERVE_HPG_NBR_COL] + df[RESERVE_AIR_NBR_COL]
    df = df.drop([RESERVE_HPG_NBR_COL, RESERVE_AIR_NBR_COL], axis=1)

    return df


def rolling_agg(
    s: pd.Series,
    window: int,
    agg: str,
    open_usually: pd.Series | None = None,
    **agg_kwargs,
) -> pd.Series:
    s = s.shift(1)

    if open_usually is not None:
        ou = open_usually.loc[s.index]
        mask_replace = (ou == 0) & (s == 0)
        s_nonzero = s.where(~mask_replace, np.nan)
    else:
        s_nonzero = s

    res = s_nonzero.rolling(window, min_periods=1).agg(agg, **agg_kwargs)

    return res


def add_basic_stats(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    aggs: list = None,
    windows: list = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 28]

    df = df.copy().sort_values([id_col, VISIT_DATE_COL])

    if aggs is None:
        aggs = [
            ("mean", {}),
            ("median", {}),
            ("std", {"ddof": 0}),
        ]

    open_s = df[OPEN_USUALLY_COL] if OPEN_USUALLY_COL in df.columns else None
    grouping = df.groupby(id_col)[target_col]
    
    new_cols = {}
    for window in windows:
        for agg_name, agg_kwargs in aggs:
            col_name = f"{target_col}_{agg_name}_{window}"
            new_cols[col_name] = grouping.transform(
                lambda x: rolling_agg(x, window, agg_name, open_s, **agg_kwargs)
            )
    
    df = df.assign(**new_cols)

    return df


def add_neighbors_stats(
    df: pd.DataFrame,
    target_col: str,
    grouping_col: str,
    aggs: list = None,
    rename_col: bool = True
):
    orig_df = df
    df = df.copy()[[VISIT_DATE_COL, grouping_col, target_col, OPEN_USUALLY_COL]]
    if rename_col:
        nbr_target = target_col + "_nbrs"
        df.rename(columns={target_col: nbr_target}, inplace=True)
    else:
        nbr_target = target_col
        orig_df = orig_df.drop(columns=[target_col])

    mask_replace = (df[OPEN_USUALLY_COL] == 0) & (df[nbr_target] == 0)
    df[nbr_target] = df[nbr_target].where(~mask_replace, np.nan)
    df = df.drop(columns=[OPEN_USUALLY_COL])

    means = df.groupby([VISIT_DATE_COL, grouping_col])[nbr_target].mean()
    res = add_basic_stats(means.reset_index(), nbr_target, grouping_col, aggs)

    df = orig_df.merge(res, on=[VISIT_DATE_COL, grouping_col], how="left")
    return df


def add_last_month_visitors(
    df: pd.DataFrame,
    target_col,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feature_col = target_col + "_last_month"

    df = df.copy()
    ref_df = df if reference_df is None else reference_df
    lookup_date_col = "_last_month_date"

    df[lookup_date_col] = df[VISIT_DATE_COL] - pd.DateOffset(months=1)
    lookup = ref_df[[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL, target_col]].rename(
        columns={VISIT_DATE_COL: lookup_date_col, target_col: feature_col}
    )

    df = df.merge(lookup, on=[AIR_RESTAURANT_ID_COL, lookup_date_col], how="left")
    df.drop(columns=lookup_date_col, inplace=True)

    return df


def add_historical_dow_mean(
    df: pd.DataFrame,
    target_col: str,
    feature_name: str,
) -> pd.DataFrame:

    df = df.copy()
    df_sorted = df.sort_values([AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])
    grouping_keys = [AIR_RESTAURANT_ID_COL, DAY_OF_WEEK_COL]
    
    def shift_and_expanding_mean(series):
        shifted = series.shift(1)
        return shifted.expanding(min_periods=1).mean()
    
    df_sorted[feature_name] = df_sorted.groupby(grouping_keys)[target_col].transform(
        shift_and_expanding_mean
    )

    return df_sorted.sort_index()


def add_reserves_difference(
    df: pd.DataFrame, visitors_col: str, reserve_col: str, feature_col: str
) -> pd.DataFrame:
    df = df.copy()
    diff = df.groupby(AIR_RESTAURANT_ID_COL)[[visitors_col, reserve_col]].shift(1)
    df[feature_col] = diff[reserve_col] - diff[visitors_col]

    return df


def add_lags(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    lags: tuple[int, ...],
    use_nbrs: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values([id_col, VISIT_DATE_COL])

    if use_nbrs:
        df_gr = (
            df.groupby([id_col, VISIT_DATE_COL])[target_col]
            .mean()
            .to_frame()
        )
        grouped = df_gr.groupby(level=0)[target_col]
        
        new_cols = {
            f"{target_col}_lag_{lag}": grouped.shift(lag)
            for lag in lags
        }
        df_gr = df_gr.assign(**new_cols)
        df_gr = df_gr.reset_index().drop(target_col, axis=1)
        df = df.merge(df_gr, on=[id_col, VISIT_DATE_COL], how="left")
    else:
        grouped = df.groupby(id_col)[target_col]
        
        new_cols = {
            f"{target_col}_lag_{lag}": grouped.shift(lag)
            for lag in lags
        }
        df = df.assign(**new_cols)

    return df
