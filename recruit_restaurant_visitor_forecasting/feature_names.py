from recruit_restaurant_visitor_forecasting.config import NBRS_SUFFIX

def weekday_opened(day: str) -> str:
    return f"{day}_open"

def nbrs_col(col: str) -> str:
    return f"{col}{NBRS_SUFFIX}"

def last_month_col(col: str) -> str:
    return f"{col}_last_month"

def lag_col(col: str, lag: int) -> str:
    return f"{col}_lag_{lag}"