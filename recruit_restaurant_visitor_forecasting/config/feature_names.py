from recruit_restaurant_visitor_forecasting.config.features import NBRS_SUFFIX

def nbrs_col(col: str) -> str:
    return f"{col}{NBRS_SUFFIX}"

def lag_col(col: str, lag: int) -> str:
    return f"{col}_lag_{lag}"

def agg_window_col(col: str, agg: str, window: int | str) -> str:
    return f"{col}_{agg}_{window}"

def agg_exp_col(col: str, agg: str | int) -> str:
    return f"{col}_{agg}"
