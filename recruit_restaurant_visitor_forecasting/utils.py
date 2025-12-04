import pandas as pd
import statsmodels.api as sm
from recruit_restaurant_visitor_forecasting.config import AIR_DAILY_COL, HPG_DAILY_COL, VISIT_DATE_COL


def get_dfs_daily_corr(
    air_df: pd.DataFrame,
    hpg_df: pd.DataFrame,
    visit_col: str,
    visitors_col: str,
    exclude_dates: list = None
) -> tuple[float, pd.DataFrame]:

    air_daily = air_df.reset_index().groupby(visit_col)[visitors_col].mean()
    hpg_daily = hpg_df.reset_index().groupby(visit_col)[visitors_col].mean()

    combined = pd.concat(
        [air_daily.rename(AIR_DAILY_COL), hpg_daily.rename(HPG_DAILY_COL)], axis=1
    )

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
        | (merged_df[visitors_col].isna())
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
