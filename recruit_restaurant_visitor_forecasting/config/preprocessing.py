import pandas as pd

from recruit_restaurant_visitor_forecasting.config.config import (
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_COL,
)
from recruit_restaurant_visitor_forecasting.config.features import (
    VISITORS_NBR_COL,
    GOLDEN_WEEK_FLG,
    DAY_OF_WEEK_COL,
    CITY_COL,
    RES_VISITORS_DIFF_NBR_COL,
    OPENED_RECENTLY_FLG,
    MEAN_PREF,
    MEDIAN_PREF,
    STD_PREF,
    MAX_PREF,
    MIN_PREF,
    OPEN_USUALLY_COL,
    TOTAL_RES_COL,
    TOTAL_RES_NBR_COL,
    VISITORS_LAST_MONTH,
    VISITORS_DOW,
)
from recruit_restaurant_visitor_forecasting.config.feature_names import (
    agg_window_col,
    lag_col,
    agg_exp_col,
)

DROP_FEATURES = [
    *[agg_window_col(VISITORS_COL, MEAN_PREF, n) for n in [7, 28]],
    *[agg_window_col(VISITORS_COL, MEDIAN_PREF, n) for n in [7, 14]],
    *[agg_window_col(VISITORS_COL, STD_PREF, n) for n in [7, 14]],
    *[agg_window_col(VISITORS_COL, MAX_PREF, n) for n in [14, 28]],
    *[agg_window_col(VISITORS_COL, MIN_PREF, n) for n in [14, 28]],
    *[agg_window_col(VISITORS_NBR_COL, MEAN_PREF, n) for n in [7, 14, 28]],
    *[agg_window_col(VISITORS_NBR_COL, MEDIAN_PREF, n) for n in [7, 28]],
    *[agg_window_col(VISITORS_NBR_COL, STD_PREF, n) for n in [7, 14]],
    *[agg_window_col(RES_VISITORS_DIFF_NBR_COL, MEAN_PREF, n) for n in [7, 14, 28]],
    RES_VISITORS_DIFF_NBR_COL,
    GOLDEN_WEEK_FLG,
    OPENED_RECENTLY_FLG,
]
REMAINING_FEATURES = [
    GOLDEN_WEEK_FLG,
    OPEN_USUALLY_COL,
    TOTAL_RES_COL,
    TOTAL_RES_NBR_COL,
    VISITORS_LAST_MONTH,
    agg_window_col(VISITORS_COL, MEDIAN_PREF, 7),
    agg_window_col(VISITORS_COL, STD_PREF, 7),
    agg_window_col(VISITORS_COL, MAX_PREF, 28),
    agg_window_col(VISITORS_DOW, STD_PREF, 4),
    lag_col(VISITORS_NBR_COL, 7),
    *[agg_window_col(VISITORS_COL, MIN_PREF, n) for n in [7, 14, 28]],
    *[agg_exp_col(VISITORS_DOW, agg) for agg in [MEDIAN_PREF, STD_PREF]],
    *[lag_col(VISITORS_COL, lag) for lag in [1, 7, 28]],
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
VARIANCE_FS_THRESHOLD = 0.01
TARGET_CORR_THRESHOLD = 0.2
