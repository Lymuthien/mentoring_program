from pathlib import Path

import numpy as np
import matplotlib as mpl
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

# Df names

AIR_VISIT_STR = "Air visit"
AIR_RESERVE_STR = "Air reserve"
HPG_RESERVE_STR = "Hpg reserve"
AIR_STORE_STR = "Air store"
HPG_STORE_STR = "Hpg store"
PERCENTAGE_COL = "percentage"
SAMPLE_SUBMISSION_STR = "Sample submission"

# Column names

AIR_RESTAURANT_ID_COL = "air_store_id"
HPG_RESTAURANT_ID_COL = "hpg_store_id"
VISIT_DATETIME_COL = "visit_datetime"
RESERVE_DATETIME_COL = "reserve_datetime"
VISIT_DATE_COL = "visit_date"
CALENDAR_DATE_COL = "calendar_date"
RESERVE_VISITORS_COL = "reserve_visitors"
VISITORS_COL = "visitors"
AIR_AREA_COL = "air_area_name"
HPG_AREA_COL = "hpg_area_name"
LATITUDE_COL = "latitude"
LONGITUDE_COL = "longitude"
AIR_GENRE_COL = "air_genre_name"
HPG_GENRE_COL = "hpg_genre_name"
AIR_STORE_REL_STR = AIR_STORE_STR + " relation"
HPG_STORE_REL_STR = HPG_STORE_STR + " relation"
CITY_REGION_COL = "city_region"
CITY_COL = "city"
VISITORS_DIFF_COL = "visitors_difference"
AIR_DAILY_COL = "air_daily"
HPG_DAILY_COL = "hpg_daily"
RESERVE_HPG = RESERVE_VISITORS_COL + "_hpg"
RESERVE_AIR = RESERVE_VISITORS_COL + "_air"
DAYS_FROM_HOL_COL = "days_from_holiday"

# Seasonal columns

YEAR_COL = "year"
MONTH_COL = "month"
WEEK_COL = "week"
DAY_OF_WEEK_COL = "day_of_week"
DAY_STR_COL = "day_str"
YEAR_MONTH_COL = "year_month"
DAY_COL = "day"

np.random.seed(42)

MONTH_COLORS = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), 12, replace=False)
