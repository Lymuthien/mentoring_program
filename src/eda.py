from IPython.display import display
import pandas as pd

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
