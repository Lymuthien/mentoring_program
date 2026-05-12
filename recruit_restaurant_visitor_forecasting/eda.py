from IPython.display import display
import pandas as pd

from recruit_restaurant_visitor_forecasting.config.config import VISITORS_COL
from recruit_restaurant_visitor_forecasting.config.features import OPEN_USUALLY_COL
from recruit_restaurant_visitor_forecasting.plots import plot_pairs_rel

type DatasetInfo = tuple[pd.DataFrame, str, str]
type UniqueIds = dict[str, set]


def compute_unique(datasets: list[DatasetInfo]) -> UniqueIds:
    result = {}
    for df, name, col in datasets:
        ids = set(df[col].unique())
        result[name] = ids
        print(f"{name} unique restaurants ids: {len(ids)}")
    return result


def print_intersection(ids: UniqueIds, name1: str, name2: str):
    set1 = ids[name1]
    set2 = ids[name2]
    print(f"{name1} & {name2} intersection: {len(set1 & set2)}")


def print_df_len(df: pd.DataFrame, df_str: str):
    print(df_str, "count of rows:", df.shape[0])


def print_problematic_rows(problematic_rows: pd.DataFrame):
    print(f"Found {len(problematic_rows)} problematic rows")
    display(problematic_rows.head())


def print_low_frequency_values(df: pd.DataFrame, col: str, frequency: int) -> None:
    counts = df[col].value_counts()
    print(
        f"Values with frequency less than {frequency}: ",
        len(counts[counts <= frequency]),
    )


def print_ou_vis_corr(corr: float):
    print(f"Correlation with visitors: {corr:.2f}")


def print_means(lower_mean: float, upper_mean: float):
    print(
        f"Visitor mean on closing days: {lower_mean:.2f}.\n"
        f"Visitor mean on opening days: {upper_mean:.2f}."
    )


def show_open_usually_stats(df: pd.DataFrame, prob_threshold: float = 0.5):
    corr = df[VISITORS_COL].corr(df[OPEN_USUALLY_COL])

    print_ou_vis_corr(corr)
    plot_pairs_rel([(OPEN_USUALLY_COL, VISITORS_COL)], df)

    lower = df[df[OPEN_USUALLY_COL] <= prob_threshold][VISITORS_COL].mean()
    upper = df[df[OPEN_USUALLY_COL] > prob_threshold][VISITORS_COL].mean()
    print_means(lower, upper)


def print_period_problematic_rest(period: str, rest_count: int, all_zero_count: int):
    print("Period:", period)
    print("[Restaurants with reservation possibility are taken into account]")
    print("Count of restaurants:", rest_count)
    print("Count of restaurants with all 0 reservations:", all_zero_count)


def print_best_params(optuna_study):
    print("Best parameters:", optuna_study.best_params)
    print(f"Best score: {optuna_study.best_value:.5f}")