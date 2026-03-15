from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Column names

AIR_RESTAURANT_ID_COL = "air_store_id"
HPG_RESTAURANT_ID_COL = "hpg_store_id"
VISIT_DATE_COL = "visit_date"
VISIT_DATETIME_COL = "visit_datetime"
RESERVE_DATETIME_COL = "reserve_datetime"
CALENDAR_DATE_COL = "calendar_date"
RESERVE_VISITORS_COL = "reserve_visitors"
VISITORS_COL = "visitors"
AIR_AREA_COL = "air_area_name"
HPG_AREA_COL = "hpg_area_name"
LATITUDE_COL = "latitude"
LONGITUDE_COL = "longitude"
AIR_GENRE_COL = "air_genre_name"
HPG_GENRE_COL = "hpg_genre_name"
HOLIDAY_COL = "holiday_flg"

# Seasonal columns

YEAR_COL = "year"
MONTH_COL = "month"
WEEK_COL = "week"
DAY_OF_WEEK_COL = "day_of_week"
DAY_STR_COL = "day_str"
YEAR_MONTH_COL = "year_month"
DAY_COL = "day"

# Feature columns

NBRS_SUFFIX = "_nbrs"
VISITORS_DIFF_COL = "visitors_difference"
AIR_DAILY_COL = "air_daily"
HPG_DAILY_COL = "hpg_daily"
PERCENTAGE_COL = "percentage"
CITY_COL = "city"
CITY_REGION_COL = "city_region"
OPEN_DATE_COL = "open_date"
DAYS_FROM_HOL_COL = "days_from_holiday"
RESERVE_AIR_COL = "air_reserves"
RESERVE_HPG_COL = "hpg_reserves"
RESERVE_AIR_NBR_COL = RESERVE_AIR_COL + NBRS_SUFFIX
RESERVE_HPG_NBR_COL = RESERVE_HPG_COL + NBRS_SUFFIX
TOTAL_RES_COL = "total_reservations"
TOTAL_RES_NBR_COL = TOTAL_RES_COL + NBRS_SUFFIX
VISITORS_NBR_COL = VISITORS_COL + NBRS_SUFFIX
VISITORS_DOW_MEAN_COL = "visitors_dow_mean"
VISITORS_DOW_MEAN_NBR_COL = VISITORS_DOW_MEAN_COL + NBRS_SUFFIX
RES_VISITORS_DIFF_COL = "res_visitors_diff"
RES_VISITORS_DIFF_NBR_COL = "nbr_res_visitors_diff"
GOLDEN_WEEK_FLG = "golden_week_flg"

DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Th", "Fri", "Sat", "Sun"]

ACTUAL_MEAN = "actual_mean"
PRED_MEAN = "predicted_mean"

USELESS_FEATURES = [
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
    *USELESS_FEATURES,
]

CITY_DIV_N = 1
CITY_REGION_DIV_N = 2
