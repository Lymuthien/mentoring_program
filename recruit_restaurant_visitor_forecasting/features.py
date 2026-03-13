import numpy as np
import pandas as pd
from recruit_restaurant_visitor_forecasting.config import (
    AIR_RESTAURANT_ID_COL,
    DAY_COL,
    DAY_OF_WEEK_COL,
    DAY_STR_COL,
    DAYS_FROM_HOL_COL,
    DAYS_OF_WEEK,
    HOLIDAY_COL,
    HPG_RESTAURANT_ID_COL,
    MONTH_COL,
    OPEN_DATE_COL,
    PERCENTAGE_COL,
    RESERVE_AIR_COL,
    RESERVE_AIR_NBR_COL,
    RESERVE_HPG_COL,
    RESERVE_HPG_NBR_COL,
    RESERVE_VISITORS_COL,
    TOTAL_RESERVES_COL,
    TOTAL_RESERVES_NBR_COL,
    VISIT_DATE_COL,
    WEEK_COL,
    YEAR_COL,
    YEAR_MONTH_COL,
)
from recruit_restaurant_visitor_forecasting.feature_names import (
    weekday_opened,
    nbrs_col,
    last_month_col,
    lag_col,
)

SCALE_COL = "scale"
AVG_SCALE_COL = "avg_scale"
EST_AIR_RES_COL = "estimated_air_reserve"
GOLDEN_WEEK_FLG = "golden_week_flg"
OPENED_RECENTLY_FLG = "opened_recently"
DAYS_FROM_LAST_VISIT_COL = "days_from_last_visit"
OPEN_USUALLY_COL = "open_usually"
G_CUM_SUM = "global_cum_sum"
G_CUM_CNT = "global_cum_count"
G_MEAN = "global_mean"


def get_first_str_values(s: pd.Series, n: int, sep: str = " ") -> pd.Series:
    return s.str.split(sep).str[:n].str.join(sep)


def get_opened_restaurants_pct(
    df: pd.DataFrame, visit_col: str, id_col: str
) -> pd.DataFrame:
    min_dates = df.groupby(id_col)[visit_col].min()
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
    grouped = df.groupby([AIR_RESTAURANT_ID_COL, df[visit_col].dt.dayofweek])
    count = grouped.size()
    nonzero_count = grouped[visitors_col].agg(lambda s: s.ne(0).sum())

    pct_df = (nonzero_count / count).unstack(fill_value=0)
    cols_map = {i: day for i, day in enumerate(DAYS_OF_WEEK)}
    pct_df = pct_df.rename(columns=cols_map).reset_index()

    return pct_df


def get_open_status(pct_df: pd.DataFrame, threshold_ratio: float = 0.5):
    pct_df = pct_df
    max_pct = pct_df[DAYS_OF_WEEK].max(axis=1)
    threshold = threshold_ratio * max_pct

    open_cols = {
        weekday_opened(day): (pct_df[day] >= threshold).astype(int)
        for day in DAYS_OF_WEEK
    }
    pct_df = pct_df.assign(**open_cols)
    pct_df = pct_df.drop(DAYS_OF_WEEK, axis=1)

    return pct_df


def add_open_usually(df: pd.DataFrame, drop_days: bool = True):
    df = df.copy()

    def get_open(row):
        day = row[DAY_OF_WEEK_COL]
        return row[weekday_opened(day)]

    df[OPEN_USUALLY_COL] = df.apply(get_open, axis=1)

    if drop_days:
        open_flags = [weekday_opened(day) for day in DAYS_OF_WEEK]
        df = df.drop(open_flags, axis=1)
    return df


def add_seasonal_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dates = df[date_col].dt

    df[YEAR_COL] = dates.year
    df[MONTH_COL] = dates.month
    df[DAY_COL] = dates.day
    df[WEEK_COL] = dates.isocalendar().week
    df[DAY_OF_WEEK_COL] = dates.day_of_week
    df[DAY_STR_COL] = dates.strftime("%a")
    df[YEAR_MONTH_COL] = dates.year.astype(str) + "_" + dates.month.astype(str)

    return df


def add_holiday_columns(df: pd.DataFrame, date_col: str):
    df = df.sort_values(date_col).reset_index(drop=True)

    inf = np.iinfo(np.int64).max
    holiday_mask = df[HOLIDAY_COL] == 1
    non_holiday_mask = ~holiday_mask
    result = np.zeros(len(df), dtype=np.int64)

    holiday_indices = df.index[holiday_mask].values
    non_holiday_indices = df.index[non_holiday_mask].values
    next_holiday_pos = np.searchsorted(holiday_indices, non_holiday_indices)
    prev_holiday_pos = next_holiday_pos - 1

    valid_prev = prev_holiday_pos >= 0
    days_since_prev = np.full(len(non_holiday_indices), inf, dtype=np.int64)
    days_since_prev[valid_prev] = (
        non_holiday_indices[valid_prev] - holiday_indices[prev_holiday_pos[valid_prev]]
    )

    valid_next = next_holiday_pos < len(holiday_indices)
    days_until_next = np.full(len(non_holiday_indices), inf, dtype=np.int64)
    days_until_next[valid_next] = (
        holiday_indices[next_holiday_pos[valid_next]] - non_holiday_indices[valid_next]
    )

    before_mask = days_until_next < days_since_prev
    result[non_holiday_mask] = np.where(before_mask, -days_until_next, days_since_prev)
    df[DAYS_FROM_HOL_COL] = result.astype(np.int64)

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
    df = df.merge(open_dates, left_on=id_col, right_index=True, how="left")

    threshold = df[date_col] - pd.DateOffset(months=n_month)
    df[OPENED_RECENTLY_FLG] = (df[OPEN_DATE_COL] >= threshold).astype(int)

    return df


def add_days_since_last_record(
    df: pd.DataFrame, id_col: str, date_col: str, history_df: pd.DataFrame = None
) -> pd.DataFrame:
    df = df.copy()
    IS_CURRENT_DF = "_is_current"

    if history_df is not None:
        history_df = history_df.copy()
        df[IS_CURRENT_DF] = True
        history_df[IS_CURRENT_DF] = False
        combined = pd.concat([history_df, df])
    else:
        combined = df
        combined[IS_CURRENT_DF] = True

    combined.sort_values([id_col, date_col], inplace=True)

    is_open = combined[OPEN_USUALLY_COL]
    prev_is_open = is_open.groupby(combined[id_col]).shift(1).fillna(0)
    seg = prev_is_open.groupby(combined[id_col]).cumsum()
    position = combined.groupby([combined[id_col], seg]).cumcount().astype(int)

    first_is_open = (
        is_open.groupby([combined[id_col], seg]).transform("first").astype(bool)
    )
    base = np.where(first_is_open, 0, 1)

    combined[DAYS_FROM_LAST_VISIT_COL] = base + position

    result = combined[combined[IS_CURRENT_DF]].drop(IS_CURRENT_DF, axis=1)

    return result.reset_index(drop=True)


def add_time_based_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category_col: str,
    target_col: str,
    feature_name: str,
    date_col: str = VISIT_DATE_COL,
    min_samples: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = (train_df[OPEN_USUALLY_COL] == 1) | (train_df[target_col] > 0)
    stats_df = train_df[mask]

    daily = stats_df.groupby(date_col)[target_col].agg(["sum", "count"]).reset_index()
    daily[G_CUM_SUM] = daily["sum"].cumsum() - daily["sum"]
    daily[G_CUM_CNT] = daily["count"].cumsum() - daily["count"]
    daily[G_MEAN] = daily[G_CUM_SUM] / daily[G_CUM_CNT].replace(0, np.nan)

    agg = (
        stats_df.groupby([category_col, date_col])[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    agg["cum_sum"] = agg.groupby(category_col)["sum"].cumsum() - agg["sum"]
    agg["cum_cnt"] = agg.groupby(category_col)["count"].cumsum() - agg["count"]

    agg = agg.merge(
        daily[[date_col, G_MEAN]],
        on=date_col,
        how="left",
    )
    category_mean = agg["cum_sum"] / agg["cum_cnt"].replace(0, np.nan)
    smoothing = 1 / (1 + np.exp(-(agg["cum_cnt"] - min_samples)))

    agg[feature_name] = (
        agg[G_MEAN] * (1 - smoothing)
        + category_mean.fillna(agg[G_MEAN]) * smoothing
    )

    first_dates = agg.groupby(category_col)[date_col].min().reset_index()
    first_dates.columns = [category_col, "first_date"]

    min_date = daily[date_col].min()
    max_date = daily[date_col].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

    all_categories = agg[category_col].unique()
    full_index = []
    for cat in all_categories:
        first_date = first_dates[first_dates[category_col] == cat]["first_date"].iloc[0]
        cat_dates = all_dates[all_dates >= first_date]
        for d in cat_dates:
            full_index.append({category_col: cat, date_col: d})

    full_df = pd.DataFrame(full_index)

    enc_full = full_df.merge(
        agg[[category_col, date_col, feature_name]],
        on=[category_col, date_col],
        how="left",
    )

    enc_full[feature_name] = enc_full.groupby(category_col)[feature_name].ffill()

    enc = (
        enc_full[[category_col, date_col, feature_name]]
        .sort_values([category_col, date_col])
        .reset_index(drop=True)
    )

    last_genre_values = enc.loc[
        enc.groupby(category_col)[date_col].idxmax(), [category_col, feature_name]
    ].reset_index(drop=True)
    last_global_mean = daily[G_MEAN].iloc[-1] if len(daily) > 0 else 0

    train_encoded = train_df.merge(enc, on=[category_col, date_col], how="left")

    test_encoded = test_df.merge(
        last_genre_values, on=[category_col], how="left"
    ).fillna(last_global_mean)

    return train_encoded, test_encoded


def add_sum_of_reserves(
    df: pd.DataFrame, output_col: str = None, id_col: str = AIR_RESTAURANT_ID_COL
) -> pd.DataFrame:
    df = df.groupby([id_col, VISIT_DATE_COL])[RESERVE_VISITORS_COL].sum()
    if output_col:
        df = df.rename(output_col)

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
    df: pd.DataFrame,
    air_res: pd.DataFrame,
    hpg_res: pd.DataFrame,
    region_col: str,
) -> pd.DataFrame:
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
    df: pd.DataFrame, hpg_res: pd.DataFrame, region_col: str
) -> pd.DataFrame:
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


def prepare_masked_series(
    df: pd.DataFrame, target_col: str, open_col: str, id_col: str
) -> pd.Series:
    s = df.groupby(id_col)[target_col].shift(1)

    if open_col in df.columns:
        ou = df[open_col]
        mask = (ou == 0) & (s == 0)
        s = s.mask(mask)

    return s


def add_basic_stats(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    aggs: list = None,
    windows: list = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [7, 14, 28]

    if aggs is None:
        aggs = [
            ("mean", {}),
            ("median", {}),
            ("std", {"ddof": 0}),
        ]

    df = df.sort_values([id_col, VISIT_DATE_COL])

    masked = prepare_masked_series(
        df,
        target_col,
        OPEN_USUALLY_COL,
        id_col,
    )

    grouped = masked.groupby(df[id_col])
    result = df

    for window in windows:
        r = grouped.rolling(window, min_periods=1)

        for agg, agg_kwargs in aggs:
            col = f"{target_col}_{agg}_{window}"
            result[col] = r.agg(agg, **agg_kwargs).reset_index(level=0, drop=True)

    return result


def add_neighbors_stats(
    df: pd.DataFrame,
    target_col: str,
    grouping_col: str,
    aggs: list = None,
    rename_col: bool = True,
):
    orig_df = df
    df = df[[VISIT_DATE_COL, grouping_col, target_col, OPEN_USUALLY_COL]]
    if rename_col:
        nbr_target = nbrs_col(target_col)
        df.rename(columns={target_col: nbr_target}, inplace=True)
    else:
        nbr_target = target_col
        orig_df = orig_df.drop(columns=[target_col])

    df.loc[(df[OPEN_USUALLY_COL] == 0) & (df[nbr_target] == 0), nbr_target] = np.nan

    df = df.drop(columns=[OPEN_USUALLY_COL])

    means = df.groupby([VISIT_DATE_COL, grouping_col])[nbr_target].mean()
    res = add_basic_stats(means.reset_index(), nbr_target, grouping_col, aggs)

    cols_to_replace = res.columns.difference([VISIT_DATE_COL, grouping_col])
    orig_df = orig_df.drop(columns=cols_to_replace, errors="ignore")

    df = orig_df.join(
        res.set_index([VISIT_DATE_COL, grouping_col]), on=[VISIT_DATE_COL, grouping_col]
    )

    return df


def add_last_month_visitors(
    df: pd.DataFrame,
    target_col,
    reference_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feature_col = last_month_col(target_col)

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
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values([AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])
    grouping_keys = [AIR_RESTAURANT_ID_COL, DAY_OF_WEEK_COL]

    def shift_and_expanding_mean(series):
        shifted = series.shift(1)
        return shifted.expanding(min_periods=1).mean()

    df[feature_name] = df.groupby(grouping_keys)[target_col].transform(
        shift_and_expanding_mean
    )

    last_values = df.loc[
        df.groupby(grouping_keys)[VISIT_DATE_COL].idxmax(),
        [*grouping_keys, feature_name],
    ].reset_index(drop=True)
    test_df = test_df.merge(last_values, on=grouping_keys, how="left")

    return df.sort_index(), test_df


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
    df = df.sort_values([id_col, VISIT_DATE_COL])

    if use_nbrs:
        df_gr = df.groupby([id_col, VISIT_DATE_COL], sort=False)[target_col].mean()
        grouped = df_gr.groupby(level=0)

        for lag in lags:
            shifted = grouped.shift(lag)
            df[lag_col(target_col, lag)] = shifted.reindex(
                pd.MultiIndex.from_frame(df[[id_col, VISIT_DATE_COL]])
            ).values
    else:
        grouped = df.groupby(id_col)[target_col]

        for lag in lags:
            df[lag_col(target_col, lag)] = grouped.shift(lag).values

    return df


def drop_first_month(df: pd.DataFrame) -> pd.DataFrame:
    open_dates = df[OPEN_DATE_COL]
    mask = df[VISIT_DATE_COL] >= (open_dates + pd.DateOffset(months=1))

    return df[mask]


def calc_air_hpg_scale(
    air_df: pd.DataFrame,
    hpg_df: pd.DataFrame,
    gap_dates,
) -> tuple[pd.DataFrame, float]:
    air_non_gap = air_df[~air_df[VISIT_DATE_COL].isin(gap_dates)]
    hpg_non_gap = hpg_df[~hpg_df[VISIT_DATE_COL].isin(gap_dates)]

    air_daily = add_sum_of_reserves(air_non_gap, "air")
    hpg_daily = add_sum_of_reserves(hpg_non_gap, "hpg")
    merged = air_daily.merge(hpg_daily, on=[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])

    merged["hpg"] = merged["hpg"].replace(0, 1)
    merged[SCALE_COL] = merged["air"] / merged["hpg"]

    scaling_factors = (
        merged.groupby(AIR_RESTAURANT_ID_COL)[SCALE_COL]
        .mean()
        .rename(AVG_SCALE_COL)
        .reset_index()
    )
    overall_median = merged[SCALE_COL].median()

    return scaling_factors, overall_median


def fill_air_res_without_hpg(
    air_df: pd.DataFrame,
    gap_dates: pd.Series,
    ids_with_hpg: set[str],
) -> pd.DataFrame:
    air_ids = set(air_df[AIR_RESTAURANT_ID_COL].unique())
    ids_without_hpg = air_ids - ids_with_hpg

    air_non_gap = air_df[~air_df[VISIT_DATE_COL].isin(gap_dates)]
    air_non_gap = air_non_gap[
        air_non_gap[AIR_RESTAURANT_ID_COL].isin(ids_without_hpg)
    ]

    air_non_gap[DAY_OF_WEEK_COL] = air_non_gap[VISIT_DATE_COL].dt.dayofweek
    air_daily_non_gap = (
        air_non_gap.groupby([AIR_RESTAURANT_ID_COL, DAY_OF_WEEK_COL])[
            RESERVE_VISITORS_COL
        ]
        .mean()
        .rename("dow_avg_reserve")
        .reset_index()
    )
    overall_avg = air_non_gap[RESERVE_VISITORS_COL].mean()

    all_combos = pd.DataFrame(
        {
            AIR_RESTAURANT_ID_COL: np.repeat(list(ids_without_hpg), len(gap_dates)),
            VISIT_DATE_COL: pd.to_datetime(np.tile(gap_dates, len(ids_without_hpg))),
        }
    )
    all_combos[DAY_OF_WEEK_COL] = all_combos[VISIT_DATE_COL].dt.dayofweek
    all_combos = all_combos.merge(
        air_daily_non_gap, on=[AIR_RESTAURANT_ID_COL, DAY_OF_WEEK_COL], how="left"
    )
    all_combos[EST_AIR_RES_COL] = (
        all_combos["dow_avg_reserve"].fillna(overall_avg).round().astype(int)
    )

    estimated_air_rows = pd.DataFrame(
        {
            AIR_RESTAURANT_ID_COL: all_combos[AIR_RESTAURANT_ID_COL],
            RESERVE_VISITORS_COL: all_combos[EST_AIR_RES_COL],
            VISIT_DATE_COL: all_combos[VISIT_DATE_COL],
        }
    )
    return estimated_air_rows


def fill_air_res_gaps(
    air_df: pd.DataFrame, hpg_df: pd.DataFrame, gap_dates: list[pd.Timestamp]
) -> pd.DataFrame:
    gap_dates = pd.to_datetime(gap_dates)

    scaling_factors, overall_median = calc_air_hpg_scale(air_df, hpg_df, gap_dates)

    hpg_gap = hpg_df[hpg_df[VISIT_DATE_COL].isin(gap_dates)]
    hpg_gap_daily = add_sum_of_reserves(hpg_gap)
    hpg_gap_scaled = hpg_gap_daily.merge(
        scaling_factors, on=AIR_RESTAURANT_ID_COL, how="left"
    )
    hpg_gap_scaled[AVG_SCALE_COL] = hpg_gap_scaled[AVG_SCALE_COL].fillna(overall_median)
    hpg_gap_scaled[EST_AIR_RES_COL] = (
        (hpg_gap_scaled[RESERVE_VISITORS_COL] * hpg_gap_scaled[AVG_SCALE_COL])
        .round()
        .astype(int)
    )

    estimated_air_rows_hpg = pd.DataFrame(
        {
            AIR_RESTAURANT_ID_COL: hpg_gap_scaled[AIR_RESTAURANT_ID_COL],
            RESERVE_VISITORS_COL: hpg_gap_scaled[EST_AIR_RES_COL],
            VISIT_DATE_COL: hpg_gap_scaled[VISIT_DATE_COL],
        }
    )

    est_air_rows = fill_air_res_without_hpg(
        air_df, gap_dates, set(hpg_gap_daily[AIR_RESTAURANT_ID_COL].unique())
    )
    est_air_rows = pd.concat([estimated_air_rows_hpg, est_air_rows], ignore_index=True)

    air_filtered = air_df[~air_df[VISIT_DATE_COL].isin(gap_dates)]
    air_filtered = pd.concat([air_filtered, est_air_rows], ignore_index=True)
    air_filtered = air_filtered.sort_values([AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])

    return air_filtered
