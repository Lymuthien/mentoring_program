import pandas as pd
import statsmodels.api as sm
from recruit_restaurant_visitor_forecasting.config import AIR_DAILY_COL, HPG_DAILY_COL
from IPython.display import display

type DatasetInfo = tuple[pd.DataFrame, str, str]
type UniqueIds = dict[str, set]


def compute_unique(datasets: list[DatasetInfo]) -> UniqueIds:
    result = {}
    for df, name, col in datasets:
        ids = set(df[col].unique())
        result[name] = ids
        print(f"{name} unique restaurants ids: {len(ids)}")
    return result


def print_intersection(ids: UniqueIds, name1, name2):
    set1 = ids[name1]
    set2 = ids[name2]
    print(f"{name1} & {name2} intersection: {len(set1 & set2)}")


def mean_by_group(df: pd.DataFrame, grouping_col: str, col: str):
    return df.groupby(grouping_col)[col].mean()


def get_dfs_daily_corr(
    air_df: pd.DataFrame, hpg_df: pd.DataFrame, visit_col: str, visitors_col: str
) -> tuple[float, pd.DataFrame]:
    air_daily = mean_by_group(air_df.reset_index(), visit_col, visitors_col)
    hpg_daily = mean_by_group(hpg_df.reset_index(), visit_col, visitors_col)

    combined = pd.concat(
        [air_daily.rename(AIR_DAILY_COL), hpg_daily.rename(HPG_DAILY_COL)], axis=1
    )
    corr = combined[AIR_DAILY_COL].corr(combined[HPG_DAILY_COL])

    return corr, combined


def print_df_len(df: pd.DataFrame, df_str: str):
    print(df_str, "count of rows:", df.shape[0])


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
    print_df_len(daily_visitors, "Daily visitors (sum) of grouping df")
    merged_df = pd.merge(
        merging_df,
        daily_visitors,
        on=[store_col, date_col],
        how="outer",
    )

    problematic_rows = merged_df[
        (merged_df[visitors_col] < merged_df[reserve_visitors_col])
        | (merged_df[visitors_col].isna())
    ][[date_col, store_col, visitors_col, reserve_visitors_col]]

    return problematic_rows, merged_df


def print_problematic_rows(problematic_rows: pd.DataFrame):
    print(f"Found {len(problematic_rows)} problematic rows")
    display(problematic_rows.head())


def filter_by_column_comparison(
    df: pd.DataFrame, greater_column: str, smaller_column: str
) -> pd.DataFrame:
    return df[df[greater_column] > df[smaller_column]]


def print_low_frequency_values(df: pd.DataFrame, col: str, frequency: int) -> None:
    counts = df[col].value_counts()
    print(
        f"Values with frequency less than {frequency}: ",
        len(counts[counts <= frequency]),
    )


def compute_acf(df: pd.DataFrame, col: str, nlags: int = 7) -> pd.DataFrame:
    values = df.values if isinstance(df, pd.Series) else df[col].values

    acf_vals = sm.tsa.acf(values, nlags=nlags, fft=False, bartlett_confint=False)

    return pd.DataFrame(
        {"ACF": acf_vals}, index=pd.RangeIndex(start=0, stop=nlags + 1, name="Lag")
    )[1:]
