import pandas as pd

from src.config.config import AIR_RESTAURANT_ID_COL, VISIT_DATE_COL
from src.config.features import (
    VISITORS_NBR_COL,
    GOLDEN_WEEK_FLG,
    DAY_OF_WEEK_COL,
    CITY_COL,
    RES_VISITORS_DIFF_NBR_COL,
    OPENED_RECENTLY_FLG,
    DAYS_FROM_LAST_VISIT_COL,
)

DROP_FEATURES = [
    *[
        VISITORS_NBR_COL + suffix
        for suffix in [
            "_mean_7",
            "_mean_14",
            "_mean_28",
            "_median_7",
            "_median_28",
            "_std_7",
            "_std_14",
        ]
    ],
    *[
        RES_VISITORS_DIFF_NBR_COL + suffix
        for suffix in ["_mean_7", "_mean_14", "_mean_28"]
    ],
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
