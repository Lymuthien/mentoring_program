import pandas as pd

from recruit_restaurant_visitor_forecasting.config.config import AIR_RESTAURANT_ID_COL, VISIT_DATE_COL
from recruit_restaurant_visitor_forecasting.config.features import (
    VISITORS_NBR_COL,
    GOLDEN_WEEK_FLG,
    DAY_OF_WEEK_COL,
    CITY_COL,
    RES_VISITORS_DIFF_NBR_COL,
    OPENED_RECENTLY_FLG,
    DAYS_FROM_LAST_VISIT_COL,
)
from recruit_restaurant_visitor_forecasting.config.feature_names import agg_window_col

DROP_FEATURES = [
    *[agg_window_col(VISITORS_NBR_COL, "mean", n) for n in [7, 14, 28]],
    *[agg_window_col(VISITORS_NBR_COL, "median", n) for n in [7, 28]],
    *[agg_window_col(VISITORS_NBR_COL, "std", n) for n in [7, 14]],
    *[agg_window_col(RES_VISITORS_DIFF_NBR_COL, "mean", n) for n in [7, 14, 28]],
    RES_VISITORS_DIFF_NBR_COL,
    GOLDEN_WEEK_FLG,
    OPENED_RECENTLY_FLG,
    DAYS_FROM_LAST_VISIT_COL,
]
DROP_COLUMNS = [
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_NBR_COL,
    DAY_OF_WEEK_COL,
    CITY_COL,
    *DROP_FEATURES,
]

EMPTY_DATES_RANGE = pd.date_range(start="2016-07-26", end="2016-10-26").tolist()
