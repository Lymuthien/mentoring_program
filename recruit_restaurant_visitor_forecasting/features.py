import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import BallTree
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import statsmodels.api as sm

from recruit_restaurant_visitor_forecasting.config.config import (
    AIR_RESTAURANT_ID_COL,
    HOLIDAY_COL,
    HPG_RESTAURANT_ID_COL,
    RESERVE_VISITORS_COL,
    RESERVE_DATETIME_COL,
    VISIT_DATE_COL,
    VISIT_DATETIME_COL,
    LATITUDE_COL,
    LONGITUDE_COL,
    VISITORS_COL,
)
from recruit_restaurant_visitor_forecasting.config.features import (
    TOTAL_RES_COL,
    TOTAL_RES_NBR_COL,
    RESERVE_AIR_COL,
    RESERVE_AIR_NBR_COL,
    RESERVE_HPG_COL,
    RESERVE_HPG_NBR_COL,
    GOLDEN_WEEK_FLG,
    DAYS_FROM_HOL_COL,
    DAYS_OF_WEEK,
    YEAR_COL,
    YEAR_MONTH_COL,
    MONTH_COL,
    OPEN_DATE_COL,
    PERCENTAGE_COL,
    DAY_COL,
    DAY_OF_WEEK_COL,
    DAY_STR_COL,
    WEEK_COL,
    CITY_COL,
    OPEN_USUALLY_COL,
    OPENED_RECENTLY_FLG,
    MEAN_PREF,
    MEDIAN_PREF,
    STD_PREF,
    RES_IMPOSSIBILITY_COL,
    AGG_WINDOWS,
    LAGS,
    CLOSED_FLG,
    VISITORS_LAST_MONTH,
)
from recruit_restaurant_visitor_forecasting.config.feature_names import (
    nbrs_col,
    lag_col,
    agg_window_col,
    agg_exp_col,
)

pd.set_option("mode.copy_on_write", True)

SCALE_COL = "scale"
CUM_SUM = "cum_sum"
CUM_CNT = "cum_count"
G_MEAN = "global_mean"
IS_CURRENT_DF = "_is_current"
LOOKUP_DATE = "_last_month_date"
N_RESTAURANTS = "n_restaurants"
DOW_RENAME_MAP = {i: day for i, day in enumerate(DAYS_OF_WEEK)}


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


def _get_zero_pct_by_group(grouped) -> pd.DataFrame:
    count = grouped.size()
    nonzero_count = grouped[VISITORS_COL].agg(lambda s: s.ne(0).sum())
    pct_s = nonzero_count / count

    if isinstance(pct_s.index, pd.MultiIndex):
        pct_df = pct_s.unstack(fill_value=0)
    else:
        pct_df = (nonzero_count / count).to_frame().T

    return pct_df


def _get_open_by_weekday_pct(df: pd.DataFrame, all: bool = False) -> pd.DataFrame:
    if all:
        grouped = df.groupby([df[VISIT_DATE_COL].dt.dayofweek])
    else:
        grouped = df.groupby([AIR_RESTAURANT_ID_COL, df[VISIT_DATE_COL].dt.dayofweek])
    return _get_zero_pct_by_group(grouped).rename(columns=DOW_RENAME_MAP)


def _get_open_by_holiday_pct(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby([AIR_RESTAURANT_ID_COL, HOLIDAY_COL])
    return _get_zero_pct_by_group(grouped)


def _set_dow_feature(df: pd.DataFrame, other: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.merge(other, left_on=AIR_RESTAURANT_ID_COL, right_index=True)
    df[col] = df.apply(lambda x: x[x[DAY_OF_WEEK_COL]], axis=1)
    df = df.drop(columns=DAYS_OF_WEEK)

    return df


def add_open_usually_discr(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = df.copy()

    pct_df = _get_open_by_weekday_pct(df)
    threshold = pct_df[DAYS_OF_WEEK].max(axis=1) * threshold
    pct_df = pct_df[DAYS_OF_WEEK].ge(threshold, axis=0).astype(int)

    df = df.merge(pct_df, left_on=AIR_RESTAURANT_ID_COL, right_index=True)
    df[OPEN_USUALLY_COL] = df.apply(lambda x: x[x[DAY_OF_WEEK_COL]], axis=1)
    df = df.drop(columns=DAYS_OF_WEEK)
    return df


def add_open_usually_col(df: pd.DataFrame) -> pd.DataFrame:
    dow_pct = _get_open_by_weekday_pct(df)
    dow_pct_median = dow_pct.median(axis=1).rename("median")
    holiday_pct = _get_open_by_holiday_pct(df)

    dop_pct_all = _get_open_by_weekday_pct(df, all=True)
    pct_min_gen = dow_pct - dop_pct_all.values

    df = _set_dow_feature(df, dow_pct, "dow_pct")
    df = _set_dow_feature(df, pct_min_gen, "pct_min_gen")
    df = df.merge(dow_pct_median, left_on=AIR_RESTAURANT_ID_COL, right_index=True)
    df = df.merge(holiday_pct[1], left_on=AIR_RESTAURANT_ID_COL, right_index=True)
    df = df.rename(columns={1: "hol_pct"})

    X = df[[HOLIDAY_COL, "dow_pct", "hol_pct", "median", "pct_min_gen"]].values
    y = df[VISITORS_COL].ne(0).astype(int)

    model = LogisticRegression(l1_ratio=0, class_weight="balanced")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    df[OPEN_USUALLY_COL] = model.predict_proba(X_scaled)[:, 1]
    df = df.drop(columns=["dow_pct", "hol_pct", "median", "pct_min_gen"])

    return df


def add_open_usually_col_rolling(df: pd.DataFrame) -> pd.DataFrame:
    y = df[VISITORS_COL].ne(0).astype(int)
    holiday_pct = _get_open_by_holiday_pct(df)

    df = df.merge(holiday_pct[1], left_on=AIR_RESTAURANT_ID_COL, right_index=True)
    df = df.rename(columns={1: "hol_pct"})

    df["y_rolling"] = (
        y.groupby([df[AIR_RESTAURANT_ID_COL], df[DAY_OF_WEEK_COL]])
        .rolling(4)
        .mean()
        .groupby(level=[0, 1])
        .shift(1)
        .reset_index(level=[0, 1], drop=True)
    )
    df = df.dropna(subset=["y_rolling"])
    y = y.loc[df.index]
    X = df[[HOLIDAY_COL, "y_rolling", "hol_pct"]].values

    model = LogisticRegression(l1_ratio=0, class_weight="balanced")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    df[OPEN_USUALLY_COL] = model.predict_proba(X_scaled)[:, 1]
    df = df.drop(columns=["hol_pct", "y_rolling"])

    return df


def add_open_usually_discr_rolling(
    df: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    df = df.copy()

    grouped = df.groupby([AIR_RESTAURANT_ID_COL, DAY_OF_WEEK_COL])
    vis_rolling = grouped.rolling(4)[VISITORS_COL].median()
    rolling_shift = vis_rolling.groupby(level=[0, 1]).shift(1).dropna()
    opened = (rolling_shift > threshold).astype(int)
    opened = opened.reset_index(level=[0, 1], drop=True)
    df[OPEN_USUALLY_COL] = opened

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


def add_holiday_columns(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
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


def add_time_based_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category_col: str,
    target_col: str,
    feature_name: str,
    date_col: str = VISIT_DATE_COL,
    min_samples: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opened_mask = (train_df[OPEN_USUALLY_COL] == 1) | (train_df[target_col] > 0)
    df = train_df[opened_mask]

    daily = df.groupby(date_col)[target_col].agg(["sum", "count"]).reset_index()
    daily[CUM_SUM] = daily["sum"].cumsum() - daily["sum"]
    daily[CUM_CNT] = daily["count"].cumsum() - daily["count"]
    daily[G_MEAN] = daily[CUM_SUM] / daily[CUM_CNT].replace(0, np.nan)
    daily = daily[[date_col, G_MEAN]]

    agg = (
        df.groupby([category_col, date_col])[target_col]
        .agg(["sum", "count"])
        .reset_index()
    )
    agg[CUM_SUM] = agg.groupby(category_col)["sum"].cumsum() - agg["sum"]
    agg[CUM_CNT] = agg.groupby(category_col)["count"].cumsum() - agg["count"]

    agg = agg.merge(daily, on=date_col, how="left")
    category_mean = agg[CUM_SUM] / agg[CUM_CNT].replace(0, np.nan)
    smoothing = 1 / (1 + np.exp(-(agg[CUM_CNT] - min_samples)))

    agg[feature_name] = (
        agg[G_MEAN] * (1 - smoothing) + category_mean.fillna(agg[G_MEAN]) * smoothing
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
    last_global_mean = daily[G_MEAN].iloc[-1]

    train_encoded = train_df.merge(enc, on=[category_col, date_col], how="left")

    test_encoded = test_df.merge(
        last_genre_values, on=[category_col], how="left"
    ).fillna(last_global_mean)

    return train_encoded, test_encoded


def add_sum_of_reserves(
    df: pd.DataFrame,
    output_col: str = None,
    id_col: str = AIR_RESTAURANT_ID_COL,
    max_res_diff: int = 0,
) -> pd.DataFrame:
    def get_sum(df):
        return df.groupby([id_col, VISIT_DATE_COL])[RESERVE_VISITORS_COL].sum()

    col = output_col if output_col else RESERVE_VISITORS_COL
    new_df = get_sum(df).rename(col).reset_index()

    for i in range(max_res_diff):
        mask = (
            df[RESERVE_DATETIME_COL].dt.normalize() + pd.DateOffset(days=i)
            < df[VISIT_DATE_COL]
        )
        temp = get_sum(df[mask]).rename(agg_exp_col(col, i)).reset_index()
        new_df = new_df.merge(temp, on=[id_col, VISIT_DATE_COL], how="left")

    new_df = new_df.fillna(0)
    return new_df


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


def add_total_reservations(
    df: pd.DataFrame,
    air_res: pd.DataFrame,
    hpg_res: pd.DataFrame,
    region_col: str,
    max_res_diff: int = 0,
) -> pd.DataFrame:
    merge_columns = [AIR_RESTAURANT_ID_COL, VISIT_DATE_COL, region_col]
    df = (
        df.merge(air_res, on=merge_columns, how="left")
        .merge(hpg_res, on=merge_columns, how="left")
        .drop(HPG_RESTAURANT_ID_COL, axis=1)
    )

    pairs = [(RESERVE_AIR_COL, RESERVE_HPG_COL)]
    pairs += [
        (agg_exp_col(RESERVE_AIR_COL, i), agg_exp_col(RESERVE_HPG_COL, i))
        for i in range(max_res_diff)
    ]
    cols = [col for pair in pairs for col in pair]
    df[cols] = df[cols].fillna(0)
    total_cols = {}

    for i, (air_col, hpg_col) in enumerate(pairs):
        total = TOTAL_RES_COL if i == 0 else agg_exp_col(TOTAL_RES_COL, i)

        total_cols[total] = df[air_col] + df[hpg_col]

    df = pd.concat([df, pd.DataFrame(total_cols, index=df.index)], axis=1)
    df.drop(cols, axis=1, inplace=True)

    return df


def remove_repetitions(air: pd.DataFrame, hpg: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        AIR_RESTAURANT_ID_COL,
        RESERVE_DATETIME_COL,
        RESERVE_VISITORS_COL,
        VISIT_DATETIME_COL,
        VISIT_DATE_COL,
    ]
    repeated_res = air.merge(hpg, on=merge_cols).drop(columns=[HPG_RESTAURANT_ID_COL])
    repeated_res["repeated"] = 1

    hpg = hpg.merge(repeated_res, on=merge_cols, how="left")
    hpg["repeated"] = hpg["repeated"].fillna(0)
    hpg.loc[hpg["repeated"] == 1, RESERVE_VISITORS_COL] = 0

    return hpg.drop(columns=["repeated", RES_IMPOSSIBILITY_COL])


def add_total_nbr_reservations(
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
    df[TOTAL_RES_NBR_COL] = df[RESERVE_HPG_NBR_COL] + df[RESERVE_AIR_NBR_COL]
    df[TOTAL_RES_NBR_COL] = df[TOTAL_RES_NBR_COL].fillna(0)
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
    windows: list = AGG_WINDOWS,
) -> pd.DataFrame:

    if aggs is None:
        aggs = [(MEAN_PREF, {}), (MEDIAN_PREF, {}), (STD_PREF, {"ddof": 0})]

    df = df.sort_values([id_col, VISIT_DATE_COL])

    masked = prepare_masked_series(df, target_col, OPEN_USUALLY_COL, id_col)
    grouped = masked.groupby(df[id_col])

    for window in windows:
        r = grouped.rolling(window, min_periods=1)

        for agg, agg_kwargs in aggs:
            col = agg_window_col(target_col, agg, window)
            df[col] = r.agg(agg, **agg_kwargs).reset_index(level=0, drop=True)
            df[col] = df[col].fillna(0)

    return df


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
    reference_df: pd.DataFrame | None = None,
    fillna: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    ref_df = df if reference_df is None else reference_df

    df[LOOKUP_DATE] = df[VISIT_DATE_COL] - pd.DateOffset(months=1)
    lookup = ref_df[[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL, VISITORS_COL]].rename(
        columns={VISIT_DATE_COL: LOOKUP_DATE, VISITORS_COL: VISITORS_LAST_MONTH}
    )

    df = df.merge(lookup, on=[AIR_RESTAURANT_ID_COL, LOOKUP_DATE], how="left")
    if fillna:
        df[VISITORS_LAST_MONTH] = df[VISITORS_LAST_MONTH].fillna(
            df[lag_col(VISITORS_COL, 28)]
        )
    df.drop(columns=LOOKUP_DATE, inplace=True)

    return df


def add_dow_cum_agg(
    df: pd.DataFrame,
    target_col: str,
    feature: str,
    aggs: list[str],
    test_df: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values([AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])
    grouping = [AIR_RESTAURANT_ID_COL, DAY_OF_WEEK_COL]

    exp = df.groupby(grouping)[target_col].expanding(min_periods=1)

    for agg in aggs:
        col = agg_exp_col(feature, agg)
        agg_s = exp.agg(agg).groupby(grouping).shift(1)
        df[col] = agg_s.reset_index(level=grouping, drop=True)

    if test_df is not None:
        features = [agg_exp_col(feature, agg) for agg in aggs]
        last_values = df.loc[
            df.groupby(grouping)[VISIT_DATE_COL].idxmax(), [*grouping, *features]
        ].reset_index(drop=True)
        test_df = test_df.merge(last_values, on=grouping, how="left")

    return df.sort_index(), test_df


def add_dow_rol_agg(
    df: pd.DataFrame,
    target_col: str,
    feature: str,
    aggs: list[str],
    window: int,
) -> pd.DataFrame:
    df = df.sort_values([AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])
    grouping = [AIR_RESTAURANT_ID_COL, DAY_OF_WEEK_COL]

    rol = df.groupby(grouping)[target_col].rolling(window, min_periods=1)

    for agg in aggs:
        col = agg_window_col(feature, agg, window)
        agg_s = rol.agg(agg).groupby(grouping).shift(1)
        df[col] = agg_s.reset_index(level=grouping, drop=True)

    return df.sort_index()


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
    use_nbrs: bool = False,
    lags: list[int] = LAGS,
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
    air_df: pd.DataFrame, hpg_df: pd.DataFrame, gap_dates
) -> tuple[pd.DataFrame, float]:
    air_non_gap = air_df[~air_df[VISIT_DATE_COL].isin(gap_dates)]
    hpg_non_gap = hpg_df[~hpg_df[VISIT_DATE_COL].isin(gap_dates)]

    air_daily = add_sum_of_reserves(air_non_gap, "air")
    hpg_daily = add_sum_of_reserves(hpg_non_gap, "hpg")
    merged = air_daily.merge(hpg_daily, on=[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])

    merged["hpg"] = merged["hpg"].replace(0, 1)
    merged[SCALE_COL] = merged["air"] / merged["hpg"]

    scaling = merged.groupby(AIR_RESTAURANT_ID_COL)[SCALE_COL].mean().reset_index()
    overall_median = merged[SCALE_COL].median()

    return scaling, overall_median


def fill_air_res_by_dow(
    air_df: pd.DataFrame,
    gap_dates: pd.Series,
    ids_with_hpg,
    consider_hpg: bool,
    max_date_offset: int,
    min_date_offset: int,
) -> pd.DataFrame:
    ID_COL = AIR_RESTAURANT_ID_COL
    ids = set(air_df[ID_COL].unique())
    if consider_hpg:
        ids -= ids_with_hpg

    air_ngap = air_df[~air_df[VISIT_DATE_COL].isin(gap_dates)]
    mask = air_ngap[ID_COL].isin(ids)
    mask &= air_ngap[VISIT_DATE_COL] >= gap_dates.min() - pd.DateOffset(
        days=min_date_offset
    )
    mask &= air_ngap[VISIT_DATE_COL] <= gap_dates.max() + pd.DateOffset(
        days=max_date_offset
    )
    mask &= air_ngap[RES_IMPOSSIBILITY_COL] == 0
    air_ngap = air_ngap.loc[mask]

    air_ngap[DAY_OF_WEEK_COL] = air_ngap[VISIT_DATE_COL].dt.dayofweek
    reservations = air_ngap.groupby([ID_COL, DAY_OF_WEEK_COL])[RESERVE_VISITORS_COL]
    daily_res = reservations.mean().reset_index()

    all_combos = pd.DataFrame(
        {
            ID_COL: np.repeat(list(ids), len(gap_dates)),
            VISIT_DATE_COL: pd.to_datetime(np.tile(gap_dates, len(ids))),
        }
    )
    all_combos[DAY_OF_WEEK_COL] = all_combos[VISIT_DATE_COL].dt.dayofweek
    all_combos = all_combos.merge(daily_res, on=[ID_COL, DAY_OF_WEEK_COL], how="left")
    all_combos[RESERVE_VISITORS_COL] = (
        all_combos[RESERVE_VISITORS_COL].fillna(0).round().astype(int)
    )

    est_air_rows = all_combos[[ID_COL, RESERVE_VISITORS_COL, VISIT_DATE_COL]]

    return est_air_rows


def fill_air_res_gaps(
    air_df: pd.DataFrame,
    hpg_df: pd.DataFrame,
    gap_dates: list[pd.Timestamp],
    consider_hpg: bool = True,
    max_date_offset: int = 0,
    min_date_offset: int = 1000,
) -> pd.DataFrame:
    gap_dates = pd.to_datetime(gap_dates)

    hpg_gap = hpg_df[hpg_df[VISIT_DATE_COL].isin(gap_dates)]
    hpg_gap_daily = add_sum_of_reserves(hpg_gap)
    hpg_ids = set(hpg_gap_daily[AIR_RESTAURANT_ID_COL].unique())
    est_res_dow = fill_air_res_by_dow(
        air_df, gap_dates, hpg_ids, consider_hpg, max_date_offset, min_date_offset
    )

    if consider_hpg:
        scaling_factors, overall_median = calc_air_hpg_scale(air_df, hpg_df, gap_dates)

        hpg_gap_scaled = hpg_gap_daily.merge(
            scaling_factors, on=AIR_RESTAURANT_ID_COL, how="left"
        )
        hpg_gap_scaled[SCALE_COL] = hpg_gap_scaled[SCALE_COL].fillna(overall_median)
        hpg_gap_scaled.replace({RESERVE_VISITORS_COL: {0: 1}}, inplace=True)
        scaled_res = hpg_gap_scaled[RESERVE_VISITORS_COL] * hpg_gap_scaled[SCALE_COL]
        hpg_gap_scaled[RESERVE_VISITORS_COL] = scaled_res.round().astype(int)

        est_res_scaled = hpg_gap_scaled[
            [AIR_RESTAURANT_ID_COL, RESERVE_VISITORS_COL, VISIT_DATE_COL]
        ]
        est_res_dow = pd.concat([est_res_scaled, est_res_dow], ignore_index=True)

    est_res_dow = est_res_dow.merge(
        air_df[[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL, RES_IMPOSSIBILITY_COL]],
        on=[AIR_RESTAURANT_ID_COL, VISIT_DATE_COL],
        how="left",
    )
    est_res_dow.loc[est_res_dow[RES_IMPOSSIBILITY_COL] == 1, RESERVE_VISITORS_COL] = 0
    air_filtered = air_df[~air_df[VISIT_DATE_COL].isin(gap_dates)]
    air_filtered = pd.concat([air_filtered, est_res_dow], ignore_index=True)

    return air_filtered.sort_values([AIR_RESTAURANT_ID_COL, VISIT_DATE_COL])


def fill_city_by_nearest(df: pd.DataFrame, none_val: str = "None") -> pd.DataFrame:
    df = df.copy()

    known = df[df[CITY_COL] != none_val]
    unknown = df[df[CITY_COL] == none_val]

    if len(unknown) == 0:
        return df

    known_coords = np.radians(known[[LATITUDE_COL, LONGITUDE_COL]].values)
    unknown_coords = np.radians(unknown[[LATITUDE_COL, LONGITUDE_COL]].values)

    tree = BallTree(known_coords, metric="haversine")

    _, ind = tree.query(unknown_coords, k=1)
    nearest_cities = known.iloc[ind.flatten()][CITY_COL].values
    df.loc[unknown.index, CITY_COL] = nearest_cities

    return df


def _prepare_before_cluster(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    cities = df.groupby(CITY_COL, as_index=False).agg(
        **{
            LATITUDE_COL: (LATITUDE_COL, "mean"),
            LONGITUDE_COL: (LONGITUDE_COL, "mean"),
            N_RESTAURANTS: (CITY_COL, "size"),
        }
    )

    lat0 = np.deg2rad(cities[LATITUDE_COL].mean())
    X = np.column_stack(
        [
            cities[LONGITUDE_COL].to_numpy() * np.cos(lat0),
            cities[LATITUDE_COL].to_numpy(),
        ]
    )

    return X, cities


def get_inertia(df: pd.DataFrame, ks, random_state: int = 42) -> pd.DataFrame:
    X, cities = _prepare_before_cluster(df)

    inertia = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        model.fit_predict(X, sample_weight=cities[N_RESTAURANTS])
        inertia.append({"k": k, "inertia": model.inertia_})

    return pd.DataFrame(inertia).set_index("k")


def cluster_cities(df: pd.DataFrame, k: int, random_state: int = 42) -> pd.DataFrame:
    X, cities = _prepare_before_cluster(df)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    cities["cluster_id"] = model.fit_predict(X, sample_weight=cities[N_RESTAURANTS])

    cluster_names = (
        cities.sort_values(N_RESTAURANTS, ascending=False)
        .groupby("cluster_id")[CITY_COL]
        .first()
        .to_dict()
    )
    cities["cluster_name"] = cities["cluster_id"].map(cluster_names)

    city_to_cluster = cities.set_index(CITY_COL)["cluster_name"].to_dict()
    df = df.copy()
    df[CITY_COL] = df[CITY_COL].map(city_to_cluster)

    return df


def add_reservation_impossibility(
    df: pd.DataFrame, air_res: pd.DataFrame, hpg_res: pd.DataFrame
) -> pd.DataFrame:
    ID_COL = AIR_RESTAURANT_ID_COL
    MIN_DATE = "min_date"
    MAX_DATE = "max_date"

    combined_res = pd.concat(
        [air_res[[ID_COL, VISIT_DATE_COL]], hpg_res[[ID_COL, VISIT_DATE_COL]]], axis=0
    )
    res_bounds = (
        combined_res.groupby(ID_COL)[VISIT_DATE_COL]
        .agg(["min", "max"])
        .rename(columns={"min": MIN_DATE, "max": MAX_DATE})
    )
    df = df.merge(res_bounds, on=ID_COL, how="left")

    missing_data = df[MIN_DATE].isna()
    out_of_range = (df[VISIT_DATE_COL] < df[MIN_DATE]) | (
        df[VISIT_DATE_COL] > df[MAX_DATE]
    )
    df[RES_IMPOSSIBILITY_COL] = (missing_data | out_of_range).astype(int)

    return df.drop(columns=[MIN_DATE, MAX_DATE])


def select_by_vif(
    df: pd.DataFrame, cols: list, threshold: int = 10, max_r_diff: float = 0.01
) -> tuple[list, pd.Series]:
    remaining_cols = cols.copy()
    drop_order = []
    last_dropped = None
    last_r = 0

    while True:
        X = df[remaining_cols].values
        res = sm.OLS(df[VISITORS_COL], sm.add_constant(X)).fit()
        if last_r - res.rsquared_adj > max_r_diff:
            remaining_cols.append(last_dropped)
            break

        last_r = res.rsquared_adj
        vifs = [vif(X, i) for i in range(len(remaining_cols))]
        vifs_max = max(vifs)
        if vifs_max <= threshold:
            break

        worst_idx = np.argmax(vifs)
        last_dropped = remaining_cols.pop(worst_idx)
        drop_order.append((last_dropped, vifs_max))

    return remaining_cols, pd.Series(dict(drop_order))


def get_features_by_variance_threshold(
    df: pd.DataFrame, threshold: float = 0.01
) -> pd.Series:
    df = df.select_dtypes(include=["number"])

    transformer = RobustScaler()
    scaled_data = transformer.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    features_var = scaled_df.var()

    return features_var[features_var < threshold].index.to_list()


def get_features_by_target_corr(df: pd.DataFrame, threshold: float = 0.2) -> pd.Series:
    df = df.corr(numeric_only=True)
    return df[df[VISITORS_COL] < threshold].index.to_list()


def add_closed_flg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    last_dates = (
        df[df[VISITORS_COL] != 0].groupby([AIR_RESTAURANT_ID_COL])[VISIT_DATE_COL].max()
    )
    threshold_dates = df[AIR_RESTAURANT_ID_COL].map(last_dates)
    df[CLOSED_FLG] = (df[VISIT_DATE_COL] > threshold_dates).astype(int)

    return df
