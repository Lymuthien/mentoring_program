import pandas as pd

from src.config.config import AIR_RESTAURANT_ID_COL, VISIT_DATE_COL
from src.config.features import (
    VISITORS_NBR_COL,
    GOLDEN_WEEK_FLG,
    DAY_OF_WEEK_COL,
    CITY_COL,
)

DROP_FEATURES = [
    *[
        VISITORS_NBR_COL + suffix
        for suffix in ["_mean_7", "_mean_14", "_median_7", "_median_28"]
    ],
    GOLDEN_WEEK_FLG,
]
DROP_COLUMNS = [
    AIR_RESTAURANT_ID_COL,
    VISIT_DATE_COL,
    VISITORS_NBR_COL,
    DAY_OF_WEEK_COL,
    CITY_COL,
    *DROP_FEATURES,
]

EMPTY_DATES_RANGE = pd.date_range(start='2016-07-26', end='2016-10-26').tolist()

