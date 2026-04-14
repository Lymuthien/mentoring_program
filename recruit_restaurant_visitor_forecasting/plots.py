from functools import wraps

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf as plot_acf_

from recruit_restaurant_visitor_forecasting.config.config import (
    LATITUDE_COL,
    LONGITUDE_COL,
    VISIT_DATE_COL,
    VISITORS_COL,
    AIR_GENRE_COL,
    RESERVE_VISITORS_COL,
    AIR_RESTAURANT_ID_COL,
)
from recruit_restaurant_visitor_forecasting.config.features import (
    DAY_OF_WEEK_COL,
    DAY_STR_COL,
    MONTH_COL,
    ACTUAL_MEAN,
    PRED_MEAN,
    CITY_COL,
    CITY_REGION_COL,
)
from recruit_restaurant_visitor_forecasting.utils import (
    calc_errors,
    calc_daily_errors_by_group,
    calc_daily_errors,
    calc_daily_mean_by_group,
    MEAN_ERROR,
    STD_ERROR,
)

CITY_STR = "City"
CITY_REGION_STR = "City-Region"
GENRE_STR = "Genre"
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
        df.groupby(grouping_col)[date_col].min().hist(bins=50)
    else:
        df.groupby(grouping_col)[date_col].max().hist(bins=50)
    text: str = "first" if first else "last"
    plt.xlabel(f"Date of {text} mention")
    plt.ylabel("Count of restaurants")
    plt.title(f"{df_str} - {text.capitalize()} visit mention dates - Histogram")
    plt.show()


@show_figure(figsize=(10, 6))
def plot_opened_restaurants(df: pd.Series, df_str: str) -> None:
    df.plot()
    plt.xlabel("Date")
    plt.ylabel("Count of opened (not missed) restaurants")
    plt.title(f"{df_str} - Not missed restaurants - Time plot")


@show_figure(figsize=(10, 6))
def plot_opened_restaurants_pct(df: pd.Series, df_str: str):
    df.plot()
    plt.xlabel("Date")
    plt.ylabel("Percentage of not missed restaurants (%)")
    plt.title(f"{df_str} - Percentage of mentioned restaurants - Time plot")
    plt.grid(True)


@show_figure(figsize=(14, 5))
def plot_daily_corr(df: pd.DataFrame, col1: str, col2: str) -> None:
    plt.plot(df.index, df[col1], label=col1.replace("_", " "), alpha=0.7)
    plt.plot(df.index, df[col2], label=col2.replace("_", " "), alpha=0.7)
    plt.title("Daily Mean Reservations")
    plt.legend()
    plt.grid(True)


def plot_mean_diff_by_date(df: pd.Series):
    df.plot(
        xlabel="Date",
        ylabel="Difference",
        title=f"Mean difference between reservations and visitors",
        grid=True,
    )
    plt.show()


def _subplot_mean_visitors(df: pd.Series, subplot: list, df_str: str) -> None:
    plt.subplot(*subplot)
    df.plot(linewidth=0.8, color="orange")
    plt.title(f"{df_str} - Mean visitors by visit date - Time Plot")
    plt.xlabel("Date")
    plt.ylabel("Visitors count")
    plt.grid(True, alpha=0.3)


def _subplot_visitors_hist(df: pd.Series, subplot: list, df_str: str) -> None:
    plt.subplot(*subplot)
    df.hist(bins=50, edgecolor="black")
    plt.title(f"{df_str} - Mean visitors by visit date - Histogram")
    plt.xlabel("Mean visitors count")
    plt.ylabel("Count of days")
    plt.grid(True, alpha=0.3)


def _subplot_reservations_count(df: pd.DataFrame, subplot: list, df_str: str) -> None:
    plt.subplot(*subplot)
    df.plot()
    plt.title(f"{df_str} - Count of reservations by visit date - Time plot")
    plt.xlabel("Day")
    plt.ylabel("Count of reservations")
    plt.grid(True, alpha=0.3)


def _subplot_visitors_sum(df: pd.Series, subplot: list, df_str: str) -> None:
    plt.subplot(*subplot)
    df.plot()
    plt.title(f"{df_str} - Sum of visitors by visit date - Time plot")
    plt.xlabel("Day")
    plt.ylabel("Sum of visitors")
    plt.grid(True, alpha=0.3)


def _subplot_visitors_by_restaurant(df: pd.Series, subplot: list, df_str: str) -> None:
    plt.subplot(*subplot)
    df.hist(bins=30, edgecolor="black")
    plt.title(f"{df_str} - Mean visitors by restaurant - Histogram")
    plt.xlabel("Mean visitors count")
    plt.ylabel("Count of restaurants")
    plt.grid(True, alpha=0.3)


@show_figure(figsize=(15, 10))
def plot_visitors_df_distr(df: pd.DataFrame, df_str: str):
    visitors = df.groupby(VISIT_DATE_COL)[VISITORS_COL]
    mean_visitors = visitors.mean()
    visitors_by_rst = df.groupby(AIR_RESTAURANT_ID_COL)[VISITORS_COL].mean()

    _subplot_mean_visitors(mean_visitors, [2, 2, 1], df_str)
    _subplot_visitors_hist(mean_visitors, [2, 2, 2], df_str)
    _subplot_visitors_sum(visitors.sum(), [2, 2, 3], df_str)
    _subplot_visitors_by_restaurant(visitors_by_rst, [2, 2, 4], df_str)

    plt.tight_layout()


@show_figure(figsize=(15, 10))
def plot_reserve_df_distr(df: pd.DataFrame, df_str: str):
    visitors = df.groupby(VISIT_DATE_COL)[RESERVE_VISITORS_COL]
    mean_visitors = visitors.mean()

    _subplot_mean_visitors(mean_visitors, [2, 2, 1], df_str)
    _subplot_visitors_hist(mean_visitors, [2, 2, 2], df_str)
    _subplot_visitors_sum(visitors.sum(), [2, 2, 3], df_str)
    _subplot_reservations_count(visitors.count(), [2, 2, 4], df_str)

    plt.tight_layout()


def plot_location_map(df: pd.DataFrame) -> None:
    fig = px.scatter_map(
        df,
        lat=LATITUDE_COL,
        lon=LONGITUDE_COL,
        color=CITY_COL,
        hover_name=CITY_COL,
        zoom=4,
        height=700,
    )

    fig.update_layout(mapbox_style="carto-positron")
    fig.show()


def _plot_top_freq_val(
    df: pd.DataFrame,
    df_str: str,
    col: str,
    ax: plt.Axes,
    value_name: str,
    ascending: bool = False,
    count: int = 10,
):
    counts = df[col].value_counts(ascending=ascending)
    counts.head(count).plot(kind="bar", alpha=0.8, ax=ax)
    ax.set_ylabel("Count of restaurants")
    ax.set_xlabel(value_name)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"{df_str} - {value_name} {'anti-top' if ascending else 'top'}")


def plot_top_genres_cities(
    df: pd.DataFrame, df_str: str, genre_col: str, plot_mode: str = "all"
):
    if plot_mode not in ("all", "city", "genre"):
        raise ValueError("plot_mode must be 'all' or 'city' or 'genre'")

    plots = [
        (CITY_REGION_COL, CITY_REGION_STR, False),
        (CITY_REGION_COL, CITY_REGION_STR, True),
        (CITY_COL, CITY_STR, False),
        (CITY_COL, CITY_STR, True),
        (genre_col, GENRE_STR, False),
        (genre_col, GENRE_STR, True),
    ]
    if plot_mode == "all":
        n_rows = 3
    else:
        n_rows = 1
        plots = plots[2:4] if plot_mode == "city" else plots[4:]

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, n_rows * 4))

    for ax, (col, name, asc) in zip(axes.flatten(), plots):
        _plot_top_freq_val(df, df_str, col, ax, name, asc)

    plt.tight_layout()
    plt.show()


def plot_top_values_by_col(
    df: pd.Series,
    df_str: str,
    value_col: str,
    subplot: list,
    value_name: str,
    col: str,
    ascending: bool = False,
    count: int = 10,
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


@show_figure(figsize=(12, 10))
def plot_visitors_by_genre_city(df: pd.Series, df_str: str):
    plot_top_values_by_col(df, df_str, CITY_COL, [2, 2, 1], CITY_STR, VISITORS_COL)
    plot_top_values_by_col(
        df, df_str, CITY_COL, [2, 2, 2], CITY_STR, VISITORS_COL, True
    )
    plot_top_values_by_col(
        df, df_str, AIR_GENRE_COL, [2, 2, 3], GENRE_STR, VISITORS_COL
    )
    plot_top_values_by_col(
        df, df_str, AIR_GENRE_COL, [2, 2, 4], GENRE_STR, VISITORS_COL, True
    )

    plt.tight_layout()


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


def _plot_error(axes, df: pd.DataFrame, group: str):
    axes.plot(df[VISIT_DATE_COL], df[MEAN_ERROR], linewidth=1.5, alpha=0.7)
    axes.set_xlabel("Date")
    axes.set_ylabel("Average error")
    axes.set_title(f"Average prediction error per day: {group}")
    axes.grid(True, alpha=0.3)


def _hist_error(axes, df: pd.DataFrame, group: str):
    axes.hist(df[MEAN_ERROR], alpha=0.7)
    axes.set_title(f"Error distribution: {group}")
    axes.set_xlabel("Error")
    axes.set_ylabel("Count of days with this mean error")
    axes.grid(True, alpha=0.3)


def _plot_error_acf(axes, df: pd.DataFrame, group: str):
    plot_acf_(df[MEAN_ERROR], lags=30, ax=axes, zero=False)
    axes.set_title(f"ACF of daily mean error: {group}")
    axes.grid(True, alpha=0.3)


def plot_actual_pred(axes, df: pd.DataFrame, group: str):
    axes.plot(
        df[VISIT_DATE_COL], df[ACTUAL_MEAN], label="Actual", linewidth=2, alpha=0.7
    )
    axes.plot(
        df[VISIT_DATE_COL],
        df[PRED_MEAN],
        label="Predicted",
        linewidth=2,
        alpha=0.7,
        linestyle="--",
    )
    axes.set_xlabel("Date")
    axes.set_ylabel("Average visitors")
    axes.set_title(f"{group}: Average visitors per day")
    axes.legend()
    axes.grid(True, alpha=0.3)


def plot_daily_error_overall(
    features: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series
) -> None:
    error_df = calc_errors(features, y_test, y_pred)
    daily_errors = calc_daily_errors(error_df)
    group = "all restaurants"

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(nrows=2, ncols=4, figure=fig)

    ax_ts = fig.add_subplot(gs[0, :])
    _plot_error(ax_ts, daily_errors, group)

    ax_hist = fig.add_subplot(gs[1, :2])
    _hist_error(ax_hist, daily_errors, group)

    ax_acf = fig.add_subplot(gs[1, 2:])
    _plot_error_acf(ax_acf, daily_errors, group)

    plt.tight_layout()
    plt.show()


def plot_daily_error_by_group(
    features: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, group_col: str
) -> None:
    error_df = calc_errors(features, y_test, y_pred)
    daily_errors = calc_daily_errors_by_group(error_df, group_col)
    daily_means = calc_daily_mean_by_group(error_df, group_col)
    n_cols = 4

    groups = daily_errors[group_col].unique()
    n_groups = len(groups)

    fig, axes = plt.subplots(n_groups, n_cols, figsize=(n_cols * 5, n_groups * 5))
    axes = axes.flatten()

    for idx, group in enumerate(groups):
        group_str = str(group)
        group_data = daily_errors[daily_errors[group_col] == group]
        group_mean = daily_means[daily_means[group_col] == group]

        ax_ts = axes[n_cols * idx]
        _plot_error(ax_ts, group_data, group_str)
        ax_ts.tick_params(axis="x", labelrotation=45)

        ax_hist = axes[n_cols * idx + 1]
        _hist_error(ax_hist, group_data, group_str)

        ax_acf = axes[n_cols * idx + 2]
        _plot_error_acf(ax_acf, group_data, group_str)

        ax_dm = axes[n_cols * idx + 3]
        plot_actual_pred(ax_dm, group_mean, group_str)

    plt.tight_layout()
    plt.show()


@show_figure(figsize=(8, 6))
def plot_error_mean_std(df: pd.DataFrame, group_col: str):
    groups = df[group_col]
    palette = sns.color_palette("tab10", len(groups))

    for color, group in zip(palette, groups):
        subset = df[df[group_col] == group]

        plt.scatter(
            subset[MEAN_ERROR],
            subset[STD_ERROR],
            s=150,
            color=color,
            label=str(group),
            edgecolor="black",
        )

    plt.axvline(x=0, color="gray", linestyle="--", linewidth=2, alpha=0.8)
    plt.xlabel("Mean error")
    plt.ylabel("Std error")
    plt.title(f"Errors mean/std by {group_col}")
    plt.legend(title=group_col, bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()


@show_figure(figsize=(7, 7))
def plot_error_radar(
    df: pd.DataFrame,
    group_col: str,
    categories: list[str],
    normalize: bool = False,
):
    plot_df = df.copy()
    if normalize:
        for col in categories:
            max_val = plot_df[col].abs().max()
            if max_val != 0:
                plot_df[col] = plot_df[col] / max_val

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    ax = plt.subplot(polar=True)
    palette = sns.color_palette("tab10", len(plot_df))

    for color, (_, row) in zip(palette, plot_df.iterrows()):
        values = row[categories].tolist()
        values += values[:1]

        ax.plot(angles, values, color=color, linewidth=2, label=row[group_col])
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(f"Error by {group_col} - Radar chart")
    plt.legend(bbox_to_anchor=(1.3, 1))
    plt.tight_layout()


@show_figure(figsize=(12, 6))
def plot_errors_boxplot(
    features: pd.DataFrame, y_test: pd.Series, y_pred: pd.Series, group_col: str
):
    error_df = calc_errors(features, y_test, y_pred)
    daily_errors = calc_daily_errors_by_group(error_df, group_col)

    sns.boxplot(data=daily_errors, x=group_col, y=MEAN_ERROR, showfliers=True)
    plt.xticks(rotation=45)
    plt.title(f"Error by {group_col} - Box Plot")
    plt.tight_layout()


def plot_corr_matrix(corr_df: pd.DataFrame, max_abs: float = 1):
    fig = px.imshow(
        corr_df,
        labels=dict(x="feature", y="feature", color="corr"),
        x=corr_df.columns,
        y=corr_df.columns,
        color_continuous_scale="RdBu",
        zmin=-max_abs,
        zmax=max_abs,
    )
    fig.update_layout(width=900, height=800)
    fig.show()


def plot_pairs_rel(pairs: list | tuple, df: pd.DataFrame):
    n_cols = 2 if len(pairs) != 1 else 1
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, n_rows * 3))
    axs = ax.flatten() if n_cols > 1 else [ax]

    for ax, pair in zip(axs, pairs):
        df.plot.scatter(x=pair[0], y=pair[1], alpha=0.5, ax=ax)

    plt.tight_layout()


def plot_over_reservation(df: pd.Series, ax, group: str, kind: str):
    df.plot(kind=kind, ax=ax)
    ax.set_title(f"Percentage of (reservations > visitors) per {group}")
    ax.set_ylabel(f"% of rows (reservations > visitors) within group")
