import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_log_error

from recruit_restaurant_visitor_forecasting.config.config import (
    HPG_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_COL,
    AIR_RESTAURANT_ID_COL,
    AIR_GENRE_COL,
)
from recruit_restaurant_visitor_forecasting.config.features import (
    CITY_COL,
    AIR_DAILY_COL,
    HPG_DAILY_COL,
    ACTUAL_MEAN,
    PRED_MEAN,
)
from recruit_restaurant_visitor_forecasting.features import add_sum_of_reserves

MEAN_ERROR = "mean_error"
STD_ERROR = "std_error"
MIN_ABS_ERROR = "min_abs_error"
MAX_POS_ERROR = "max_pos_error"
MAX_NEG_ERROR = "max_neg_error"
ERROR_COL = "error"


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


def _get_daily_means(dates: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"date": dates.values, "actual": values.values})
    daily_actual = data.groupby("date")["actual"].mean().reset_index()
    return daily_actual


def merge_daily_pred(
    dates: pd.DataFrame, y: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    daily_actual = _get_daily_means(dates, y)
    daily_actual.columns = [VISIT_DATE_COL, ACTUAL_MEAN]
    daily_pred = _get_daily_means(dates, y_pred)
    daily_pred.columns = [VISIT_DATE_COL, PRED_MEAN]
    daily = daily_actual.merge(daily_pred, on=VISIT_DATE_COL)
    daily = daily.sort_values(VISIT_DATE_COL)

    return daily


def get_format(orig_df, labels):
    labels = labels.to_frame(VISITORS_COL)
    labels["id"] = (
        orig_df[AIR_RESTAURANT_ID_COL] + "_" + orig_df[VISIT_DATE_COL].astype(str)
    )
    return labels.reset_index(drop=True)


def optuna_cv_results_to_df(study: optuna.Study) -> pd.DataFrame:
    records = []

    for t in study.trials:
        scores = t.user_attrs.get("cv_scores")
        if scores is None:
            continue

        std_score = float(np.std(scores))

        rec = {
            "params": t.params,
            "std_test_score": std_score,
            "mean_test_score": t.value,
        }
        for i, s in enumerate(scores):
            rec[f"split_test_{i}"] = s
        records.append(rec)

    df = pd.DataFrame(records)

    ascending = study.direction == optuna.study.StudyDirection.MINIMIZE
    df["rank_test_score"] = (
        df["mean_test_score"].rank(method="min", ascending=ascending).astype(int)
    )

    return df


def build_feature_drop_list(
    all_columns: list[str], selected_features: list[str]
) -> list[str]:
    drop_set = set(all_columns) - set(selected_features)
    return list(drop_set)


def rmsle(y_true, y_pred) -> float:
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def calc_errors(
    features: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series
) -> pd.DataFrame:
    error_df = features[[VISIT_DATE_COL, AIR_RESTAURANT_ID_COL]]
    if CITY_COL in features.columns:
        error_df[CITY_COL] = features[CITY_COL]
    if AIR_GENRE_COL in features.columns:
        error_df[AIR_GENRE_COL] = features[AIR_GENRE_COL]

    error_df[ERROR_COL] = y_pred - y_test
    error_df[PRED_MEAN] = y_pred.values
    error_df[ACTUAL_MEAN] = y_test.values

    return error_df


def calc_daily_errors(df: pd.DataFrame) -> pd.DataFrame:
    daily_errors = df.groupby(VISIT_DATE_COL)[ERROR_COL].mean().reset_index()
    daily_errors.columns = [VISIT_DATE_COL, MEAN_ERROR]
    return daily_errors.sort_values(VISIT_DATE_COL)


def calc_daily_errors_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    daily_errors = (
        df.groupby([VISIT_DATE_COL, group_col])[ERROR_COL].mean().reset_index()
    )
    daily_errors.columns = [VISIT_DATE_COL, group_col, MEAN_ERROR]
    return daily_errors.sort_values([group_col, VISIT_DATE_COL])


def calc_daily_mean_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    daily = df.groupby([VISIT_DATE_COL, group_col])
    daily_act_means = daily[ACTUAL_MEAN].mean().reset_index()
    daily_pred_means = daily[PRED_MEAN].mean().reset_index()
    daily_combined = daily_act_means.merge(
        daily_pred_means, on=[VISIT_DATE_COL, group_col]
    )
    return daily_combined


def get_daily_error_stats_table(
    features: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    group_col: str,
    acf_lags: int = 14,
) -> pd.DataFrame:
    error_df = calc_errors(features, y_test, y_pred)
    daily_errors = calc_daily_errors_by_group(error_df, group_col)
    groups = daily_errors[group_col].unique()

    rows: list[dict] = []
    for group in groups:
        group_str = str(group)
        group_data = daily_errors[daily_errors[group_col] == group]
        series = group_data[MEAN_ERROR].reset_index(drop=True)

        row = {
            f"{group_col}": group_str,
            MEAN_ERROR: round(series.mean(), 3),
            STD_ERROR: round(series.std(), 3),
            MIN_ABS_ERROR: round(series.abs().min(), 3),
            MAX_POS_ERROR: round(series.max(), 3),
            MAX_NEG_ERROR: round(np.abs(series.min()), 3),
        }

        for lag in range(1, acf_lags + 1):
            acf_val = series.autocorr(lag=lag)
            row[f"acf_{lag}"] = round(acf_val, 3)

        rows.append(row)

    group_table = pd.DataFrame(rows)
    return group_table
