from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from recruit_restaurant_visitor_forecasting.config import (
    DAY_OF_WEEK_COL,
    DAY_STR_COL,
    LATITUDE_COL,
    LONGITUDE_COL,
    MONTH_COL,
    PERCENTAGE_COL,
    VISIT_DATE_COL,
    VISITORS_DIFF_COL,
    ACTUAL_MEAN,
    PRED_MEAN,
    AIR_RESTAURANT_ID_COL,
    CITY_COL,
    AIR_GENRE_COL,
)


MONTH_COLORS = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), 12, replace=False)


def show_figure(figsize=(10, 6)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            plt.figure(figsize=figsize)
            try:
                result = func(*args, **kwargs)
                plt.show()
                return result
            except Exception as e:
                plt.close()
                raise e

        return wrapper

    return decorator


def plot_restaurant_mention_hist(
    df: pd.DataFrame,
    df_str: str,
    grouping_col: str,
    date_col: str = VISIT_DATE_COL,
    first: bool = True,
):
    if first:
        df.reset_index().groupby(grouping_col)[date_col].min().hist(bins=50)
    else:
        df.reset_index().groupby(grouping_col)[date_col].max().hist(bins=50)
    text: str = "first" if first else "last"
    plt.xlabel(f"Date of {text} mention")
    plt.ylabel("Count of restaurants")
    plt.title(f"{df_str} - {text.capitalize()} visit mention dates - Histogram")
    plt.show()


@show_figure(figsize=(10, 6))
def plot_opened_restaurants(
    df: pd.DataFrame, df_str: str, visit_col: str, store_col: str
) -> None:
    plt.figure(figsize=(10, 6))
    df.groupby(visit_col)[store_col].count().plot()
    plt.xlabel("Date")
    plt.ylabel("Count of opened (not missed) restaurants")
    plt.title(f"{df_str} - Not missed restaurants - Time plot")


@show_figure(figsize=(10, 6))
def plot_opened_restaurants_pct(df: pd.DataFrame, df_str: str):
    df[PERCENTAGE_COL].plot()
    plt.xlabel("Date")
    plt.ylabel("Percentage of not missed restaurants (%)")
    plt.title(
        f"{df_str} - Percentage of not missed restaurants (including only mentioned restaurants) - Time plot"
    )
    plt.grid(True)


@show_figure(figsize=(14, 5))
def plot_daily_corr(df: pd.DataFrame, col1: str, col2: str) -> None:
    plt.plot(df.index, df[col1], label=col1.replace("_", " "), alpha=0.7)
    plt.plot(df.index, df[col2], label=col2.replace("_", " "), alpha=0.7)
    plt.title("Daily Mean Reservations")
    plt.legend()
    plt.grid(True)


def plot_mean_by_date(df: pd.DataFrame):
    mean = df.groupby(VISIT_DATE_COL)[VISITORS_DIFF_COL].mean()
    mean.plot(
        xlabel="Date",
        ylabel="Difference",
        title=f"Mean difference between reservations and visitors",
        grid=True,
    )
    plt.show()


def subplot_visitors_over_time(df: pd.DataFrame, subplot: list, df_str: str) -> None:
    plt.subplot(*subplot)
    df.plot(linewidth=0.8, color="orange")
    plt.title(f"{df_str} - Mean visitors by visit date - Time Plot")
    plt.xlabel("Date")
    plt.ylabel("Visitors count")
    plt.grid(True, alpha=0.3)


def subplot_visitors_over_time_hist(
    df: pd.DataFrame, subplot: list, df_str: str
) -> None:
    plt.subplot(*subplot)
    df.hist(bins=50, edgecolor="black")
    plt.title(f"{df_str} - Mean visitors by visit date - Histogram")
    plt.xlabel("Mean visitors count")
    plt.ylabel("Count of days")
    plt.grid(True, alpha=0.3)


def box_subplot_visitors_over_time(
    df: pd.DataFrame, subplot: list, df_str: str
) -> None:
    plt.subplot(*subplot)
    df.plot(kind="box", vert=False)
    plt.title(f"{df_str} - Mean visitors by visit date - Box Plot")
    plt.xlabel("Mean visitors count")
    plt.grid(True, alpha=0.3)


def box_subplot_reservations_over_time(
    df: pd.DataFrame, subplot: list, df_str: str
) -> None:
    plt.subplot(*subplot)
    df.plot()
    plt.title(f"{df_str} - Count of reservations by visit date - Time plot")
    plt.xlabel("Day")
    plt.ylabel("Count of reservations")
    plt.grid(True, alpha=0.3)


def subplot_visitors_by_restaurant(
    df: pd.DataFrame, subplot: list, df_str: str
) -> None:
    plt.subplot(*subplot)
    df.hist(bins=30, edgecolor="black")
    plt.title(f"{df_str} - Mean visitors by restaurant - Histogram")
    plt.xlabel("Mean visitors count")
    plt.ylabel("Count of restaurants")
    plt.grid(True, alpha=0.3)


@show_figure(figsize=(6, 6))
def plot_location_scatter(df: pd.DataFrame, df_str: str) -> None:
    plt.scatter(df[LONGITUDE_COL], df[LATITUDE_COL], s=10)
    plt.title(f"{df_str}: store locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")


def plot_top_freq_values(
    df: pd.DataFrame,
    df_str: str,
    col: str,
    subplot: list,
    value_name: str,
    ascending: bool = False,
    count=10,
):
    plt.subplot(*subplot)
    counts = df[col].value_counts(ascending=ascending)
    counts.head(count).plot(kind="bar", alpha=0.8)
    plt.ylabel("Count of restaurants")
    plt.xlabel(value_name)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{df_str} - {value_name} {'anti-top' if ascending else 'top'}")


def plot_top_values_by_col(
    df: pd.DataFrame,
    df_str: str,
    value_col: str,
    subplot: list,
    value_name: str,
    col: str,
    ascending: bool = False,
    count=10,
):
    plt.subplot(*subplot)
    means = df.groupby(value_col)[col].mean().sort_values(ascending=ascending)
    means.head(count).plot(kind="bar", alpha=0.8)
    plt.ylabel(f"Mean {col}")
    plt.xlabel(value_name)
    plt.xticks(rotation=45, ha="right")
    plt.title(
        f"{df_str} - Mean {col} by {value_name} {'anti-top' if ascending else 'top'}"
    )


@show_figure(figsize=(10, 6))
def plot_weekly_seasonality(df: pd.DataFrame, visitors_col: str, title: str):
    df_plot = (
        df[[MONTH_COL, DAY_STR_COL, visitors_col, DAY_OF_WEEK_COL]]
        .dropna()
        .groupby([DAY_STR_COL, MONTH_COL, DAY_OF_WEEK_COL])
        .mean()[[visitors_col]]
        .reset_index()
    )
    df_plot = df_plot.sort_values(by=DAY_OF_WEEK_COL, ascending=True)

    months = sorted(df_plot[MONTH_COL].unique())

    for i, month in enumerate(months):
        month_data = df_plot[df_plot[MONTH_COL] == month]
        plt.plot(
            DAY_STR_COL,
            visitors_col,
            data=month_data,
            color=MONTH_COLORS[i],
            label=month,
        )
    plt.gca().set(ylabel=visitors_col, xlabel="Day of week")
    plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title(title)


def plot_visitors_for_first_period(
    df: pd.DataFrame,
    visitors_col: str,
    n_month: int = 3,
):
    min_date = df.index.min()
    max_allowed_date = min_date + pd.DateOffset(months=n_month)
    df.loc[(df.index <= max_allowed_date)].groupby(VISIT_DATE_COL)[
        visitors_col
    ].mean().plot(
        title=f"Visitors for first {n_month} month - Time Plot",
        figsize=(10, 6),
        color="orange",
    )

    plt.ylabel("Visitors")
    plt.xlabel("Date")
    plt.grid(alpha=0.3)
    plt.show()


@show_figure(figsize=(10, 6))
def plot_acf(acf_df: pd.DataFrame, col: str, df_str: str) -> None:
    lags = acf_df.index
    values = acf_df["ACF"]

    plt.bar(lags, values, width=0.2, alpha=0.9)

    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title(f"{df_str} - {col} - ACF")


@show_figure(figsize=(12, 10))
def plot_features_target_corr(features: pd.DataFrame, target: pd.Series):
    features = features.select_dtypes(include=[np.number])
    corr = features.corrwith(target).sort_values(key=lambda s: s.abs(), ascending=False)

    corr.plot(kind="bar", title="Correlation of predictors and target", alpha=0.8)
    plt.ylabel("Correlation with target")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


@show_figure(figsize=(12, 10))
def plot_feature_importances(model, columns):
    feature_importances = pd.DataFrame(
        {
            "feature": columns,
            "importance": model.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importances = feature_importances.set_index("feature")

    feature_importances.plot(kind="bar", title="Importance of features", alpha=0.8)
    plt.ylabel("Importance")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()


def plot_daily_pred(daily: pd.DataFrame, axs, train: bool = True):
    axs.plot(
        daily[VISIT_DATE_COL],
        daily[ACTUAL_MEAN],
        label="Actual",
        linewidth=2,
        alpha=0.7,
    )
    axs.plot(
        daily[VISIT_DATE_COL],
        daily[PRED_MEAN],
        label="Predicted",
        linewidth=2,
        alpha=0.7,
        linestyle="--",
    )
    axs.set_xlabel("Date")
    axs.set_ylabel("Average visitors")
    if train:
        axs.set_title("Train Set: Average visitors per day")
    else:
        axs.set_title("Test Set: Average visitors per day")
    axs.legend()
    axs.grid(True, alpha=0.3)


def _calc_errors(
    features: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    error_df = features[[VISIT_DATE_COL, AIR_RESTAURANT_ID_COL]].copy()
    if CITY_COL in features.columns:
        error_df[CITY_COL] = features[CITY_COL]
    if AIR_GENRE_COL in features.columns:
        error_df[AIR_GENRE_COL] = features[AIR_GENRE_COL]

    error_df["error"] = (y_pred - y_test).abs()

    return error_df


def _calc_daily_errors(error_df: pd.DataFrame) -> pd.DataFrame:
    daily_errors = error_df.groupby(VISIT_DATE_COL)["error"].mean().reset_index()
    daily_errors.columns = [VISIT_DATE_COL, "mean_error"]
    return daily_errors.sort_values(VISIT_DATE_COL)


def _calc_daily_errors_by_group(error_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    daily_errors = (
        error_df.groupby([VISIT_DATE_COL, group_col])["error"].mean().reset_index()
    )
    daily_errors.columns = [VISIT_DATE_COL, group_col, "mean_error"]
    return daily_errors.sort_values([group_col, VISIT_DATE_COL])


@show_figure(figsize=(12, 6))
def plot_daily_error_overall(
    features: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
) -> None:
    error_df = _calc_errors(features, y_test, y_pred)
    daily_errors = _calc_daily_errors(error_df)

    plt.plot(
        daily_errors[VISIT_DATE_COL],
        daily_errors["mean_error"],
        linewidth=2,
        alpha=0.7,
    )
    plt.xlabel("Date")
    plt.ylabel("Average error")
    plt.title("Average prediction error per day (all restaurants)")
    plt.grid(True, alpha=0.3)


def plot_daily_error_by_group(
    features: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    group_col: str,
    n_cols: int = 3,
) -> None:
    error_df = _calc_errors(features, y_test, y_pred)
    daily_errors = _calc_daily_errors_by_group(error_df, group_col)

    groups = daily_errors[group_col].unique()
    n_groups = len(groups)

    n_rows = int(np.ceil(n_groups / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for idx, group in enumerate(groups):
        group_data = daily_errors[daily_errors[group_col] == group]
        axes[idx].plot(
            group_data[VISIT_DATE_COL],
            group_data["mean_error"],
            linewidth=1.5,
            alpha=0.7,
        )
        axes[idx].set_xlabel("Date")
        axes[idx].set_ylabel("Average error")
        axes[idx].set_title(f"Average error per day: {group}", fontsize=10)
        axes[idx].tick_params(axis="x", labelrotation=45)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
