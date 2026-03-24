from src.config.config import VISITORS_COL

# Seasonal columns

YEAR_COL = "year"
MONTH_COL = "month"
WEEK_COL = "week"
DAY_OF_WEEK_COL = "day_of_week"
DAY_STR_COL = "day_str"
YEAR_MONTH_COL = "year_month"
DAY_COL = "day"

# Feature columns

AIR_DAILY_COL = "air_daily"
HPG_DAILY_COL = "hpg_daily"

GENRE_TE = "air_genre_te"
AREA_TE = "air_city_te"

NBRS_SUFFIX = "_nbrs"
PERCENTAGE_COL = "percentage"
GOLDEN_WEEK_FLG = "golden_week_flg"
CITY_COL = "city"
CITY_REGION_COL = "city_region"

OPEN_DATE_COL = "open_date"
OPEN_USUALLY_COL = "open_usually"
OPENED_RECENTLY_FLG = "opened_recently"

DAYS_FROM_HOL_COL = "days_from_holiday"
DAYS_FROM_LAST_VISIT_COL = "days_from_last_visit"

VISITORS_DIFF_COL = "visitors_difference"
VISITORS_DOW_MEAN_COL = "visitors_dow_mean"
VISITORS_DOW_MEAN_NBR_COL = VISITORS_DOW_MEAN_COL + NBRS_SUFFIX
VISITORS_NBR_COL = VISITORS_COL + NBRS_SUFFIX

# Reserve columns

RESERVE_AIR_COL = "air_reserves"
RESERVE_HPG_COL = "hpg_reserves"
RESERVE_AIR_NBR_COL = RESERVE_AIR_COL + NBRS_SUFFIX
RESERVE_HPG_NBR_COL = RESERVE_HPG_COL + NBRS_SUFFIX
TOTAL_RES_COL = "total_reservations"
TOTAL_RES_NBR_COL = TOTAL_RES_COL + NBRS_SUFFIX
RES_VISITORS_DIFF_COL = "res_visitors_diff"
RES_VISITORS_DIFF_NBR_COL = "nbr_res_visitors_diff"


DAYS_OF_WEEK = ["Mon", "Tue", "Wed", "Th", "Fri", "Sat", "Sun"]

ACTUAL_MEAN = "actual_mean"
PRED_MEAN = "predicted_mean"

CITY_DIV_N = 1
CITY_REGION_DIV_N = 2
