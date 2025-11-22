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

def print_intersection(ids: UniqueIds, name1, name2):
    set1 = ids[name1]
    set2 = ids[name2]
    print(f"{name1} & {name2} intersection: {len(set1 & set2)}")
