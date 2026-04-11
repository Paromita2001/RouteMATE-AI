"""
=============================================================
  RouteMATE_AI — Constants & Configuration
  File: src/utils/constants.py
=============================================================
  Single source of truth for all config values used
  across the project. Import from here — never hardcode.
=============================================================
"""

import os

# ══════════════════════════════════════════════════════════════════════════════
#  PROJECT PATHS
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data
RAW_DIR            = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR      = os.path.join(ROOT_DIR, "data", "processed")

RAW_TRAINS_DIR     = os.path.join(RAW_DIR, "trains")
RAW_SCHEDULE_DIR   = os.path.join(RAW_DIR, "schedule")
RAW_DELAY_DIR      = os.path.join(RAW_DIR, "delay")
RAW_TRAINLIST_DIR  = os.path.join(RAW_DIR, "train_list")

# Processed files
DELAY_CSV          = os.path.join(PROCESSED_DIR, "merged_delay.csv")
TRAIN_LIST_CSV     = os.path.join(PROCESSED_DIR, "train_list_clean.csv")
SCHEDULES_CSV      = os.path.join(PROCESSED_DIR, "schedules_clean.csv")
ROUTES_CSV         = os.path.join(PROCESSED_DIR, "train_routes_clean.csv")
MASTER_CSV         = os.path.join(PROCESSED_DIR, "master_merged.csv")

# Model
MODEL_DIR          = os.path.join(ROOT_DIR, "src", "models")
MODEL_PKL          = os.path.join(MODEL_DIR, "model.pkl")
SHAP_EXPLAINER_PKL = os.path.join(MODEL_DIR, "shap_explainer.pkl")

# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN CATEGORIES & TYPES
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_CATEGORIES = ["Express", "Passenger", "Superfast"]

TRAIN_TYPES = [
    "Rajdhani",
    "Humsafar",
    "Tejas",
    "Superfast",
    "Mail/Express",
    "Express",
    "Passenger",
]

TRAIN_CATEGORY_CODE = {
    "Express"   : 0,
    "Passenger" : 1,
    "Superfast" : 2,
}

IS_SUPERFAST_TYPE = {
    "Rajdhani"    : True,
    "Humsafar"    : True,
    "Tejas"       : True,
    "Superfast"   : True,
    "Mail/Express": False,
    "Express"     : False,
    "Passenger"   : False,
}

# ══════════════════════════════════════════════════════════════════════════════
#  TRAVEL CLASSES
# ══════════════════════════════════════════════════════════════════════════════

ALL_CLASSES = ["1A", "2A", "3A", "SL", "CC", "2S", "GEN"]

AC_CLASSES  = {"1A", "2A", "3A", "CC"}

CLASS_FULL_NAME = {
    "1A" : "First AC",
    "2A" : "Second AC (2-Tier)",
    "3A" : "Third AC (3-Tier)",
    "SL" : "Sleeper",
    "CC" : "AC Chair Car",
    "2S" : "Second Sitting",
    "GEN": "General / Unreserved",
}

CLASS_COACH_CAPACITY = {
    "1A" : 24,
    "2A" : 46,
    "3A" : 64,
    "SL" : 72,
    "CC" : 78,
    "2S" : 108,
    "GEN": 90,
}

# ══════════════════════════════════════════════════════════════════════════════
#  DELAY THRESHOLDS (minutes)
# ══════════════════════════════════════════════════════════════════════════════

DELAY_THRESHOLDS = {
    "on_time"     : 15,     # ≤15 min = On Time
    "slight"      : 60,     # 16–60 min = Slight Delay
    # >60 min = Significantly Delayed
}

DELAY_CATEGORIES = {
    0: "On Time",
    1: "Slight Delay",
    2: "Significantly Delayed",
}

# Risk label thresholds for predicted delay
RISK_LOW_MAX    = 20    # ≤20 min → 🟢 Low
RISK_MEDIUM_MAX = 60    # 21–60 min → 🟡 Medium
                        # >60 min → 🔴 High

RISK_LABELS = {
    "low"    : "🟢 Low",
    "medium" : "🟡 Medium",
    "high"   : "🔴 High",
}

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE SCORING
# ══════════════════════════════════════════════════════════════════════════════

# Penalty added per train change
INTERCHANGE_PENALTY_MIN = 30   # assume 30 min buffer per change

# Route score → label
SCORE_LABELS = {
    80 : "⭐⭐⭐ Excellent",
    60 : "⭐⭐ Good",
    40 : "⭐ Fair",
    0  : "⚠️ High Risk",
}

MAX_ROUTE_CHANGES = 2       # max interchanges to consider
DEFAULT_TOP_N     = 3       # default number of routes to return

# ══════════════════════════════════════════════════════════════════════════════
#  WEATHER RISK MULTIPLIERS
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_RISK_FACTOR = {
    "Clear"        : 1.0,
    "Partly Cloudy": 1.0,
    "Cloudy"       : 1.1,
    "Fog"          : 1.8,
    "Drizzle"      : 1.2,
    "Rain"         : 1.4,
    "Heavy Rain"   : 1.7,
    "Thunderstorm" : 2.0,
    "Snow"         : 2.5,
    "Haze"         : 1.3,
}

# ══════════════════════════════════════════════════════════════════════════════
#  CALENDAR / DEMAND
# ══════════════════════════════════════════════════════════════════════════════

# Peak occupancy threshold above which Tatkal is recommended
TATKAL_THRESHOLD = 1.7

# Occupancy factor → display label
OCCUPANCY_LABELS = {
    1.8: "🔴 Very High Demand",
    1.4: "🟡 High Demand",
    0.0: "🟢 Normal",
}

# ══════════════════════════════════════════════════════════════════════════════
#  FARE (IR official 2024)
# ══════════════════════════════════════════════════════════════════════════════

GST_RATE_AC         = 0.05     # 5% on AC class base fare
MINIMUM_FARE_INR    = 25       # IR minimum ticket fare

FARE_SLABS = [
    (  50,  0.57),
    ( 100,  0.54),
    ( 150,  0.51),
    ( 200,  0.48),
    ( 250,  0.46),
    ( 300,  0.44),
    ( 400,  0.42),
    ( 500,  0.40),
    ( 600,  0.38),
    ( 700,  0.36),
    ( 800,  0.35),
    ( 900,  0.34),
    (1000,  0.33),
    (1200,  0.32),
    (1500,  0.31),
    (2000,  0.30),
    (9999,  0.28),
]

CLASS_MULTIPLIER = {
    "GEN": 1.00,
    "2S" : 1.00,
    "SL" : 1.50,
    "CC" : 2.20,
    "3A" : 2.50,
    "2A" : 3.60,
    "1A" : 5.80,
}

SUPERFAST_SURCHARGE = {
    "GEN": 15, "2S": 15, "SL": 30, "CC": 45,
    "3A": 45,  "2A": 60, "1A": 75,
}

RESERVATION_CHARGE = {
    "GEN": 0, "2S": 15, "SL": 25, "CC": 40,
    "3A": 40, "2A": 50, "1A": 60,
}

TATKAL_MULTIPLIER = {
    "SL": 1.30, "CC": 1.30, "3A": 1.30,
    "2A": 1.30, "1A": 1.30, "2S": 1.10, "GEN": 1.00,
}

# ══════════════════════════════════════════════════════════════════════════════
#  ML MODEL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "train_cat_code",
    "stop_position_ratio",
    "distance_km",
    "halt_min",
    "time_of_day_min",
    "time_block_code",
    "is_overnight",
    "journey_day",
    "running_days_count",
    "is_daily",
    "total_stops",
    "station_pct_right_time",
    "station_pct_slight",
    "station_pct_sig",
    "distance_band_code",
]

# Human-readable feature names (for SHAP / LIME plots)
FEATURE_DISPLAY_NAMES = {
    "train_cat_code"        : "Train Category",
    "stop_position_ratio"   : "Journey Progress (%)",
    "distance_km"           : "Distance (km)",
    "halt_min"              : "Scheduled Halt (min)",
    "time_of_day_min"       : "Time of Day (min)",
    "time_block_code"       : "Time Block",
    "is_overnight"          : "Overnight Journey",
    "journey_day"           : "Journey Day",
    "running_days_count"    : "Days/Week Running",
    "is_daily"              : "Daily Train",
    "total_stops"           : "Total Stops",
    "station_pct_right_time": "Station On-Time %",
    "station_pct_slight"    : "Station Slight Delay %",
    "station_pct_sig"       : "Station Major Delay %",
    "distance_band_code"    : "Distance Band",
}

TIME_BLOCK_MAP = {
    "Morning"   : 0,
    "Afternoon" : 1,
    "Evening"   : 2,
    "Night"     : 3,
    "Unknown"   : 4,
}

DISTANCE_BAND_MAP = {
    "Short"  : 0,    # ≤200 km
    "Medium" : 1,    # 201–800 km
    "Long"   : 2,    # >800 km
}

# ══════════════════════════════════════════════════════════════════════════════
#  STATION COORDINATES (lat, lon, city_name)
#  Used by weather_service and map_visualizer
# ══════════════════════════════════════════════════════════════════════════════

STATION_COORDINATES = {
    "KOAA" : (22.5726,  88.3639, "Kolkata"),
    "BDC"  : (23.0333,  88.3833, "Bandel"),
    "NDAE" : (23.4000,  88.3667, "Nabadwip"),
    "KWAE" : (23.6500,  88.1333, "Katwa"),
    "AZ"   : (24.0667,  88.2667, "Azimganj"),
    "JRLE" : (24.4667,  88.0667, "Jangipur"),
    "MLDT" : (25.0000,  88.1333, "Malda"),
    "KNE"  : (26.1000,  87.9500, "Kishanganj"),
    "NJP"  : (26.7100,  88.3547, "New Jalpaiguri"),
    "NCB"  : (26.3333,  89.4667, "New Cooch Behar"),
    "NOQ"  : (26.5000,  89.8333, "New Alipurduar"),
    "KOJ"  : (26.3833,  90.2667, "Kokrajhar"),
    "NBQ"  : (26.4833,  90.5583, "New Bongaigaon"),
    "GLPT" : (26.1833,  90.6333, "Goalpara"),
    "GHY"  : (26.1445,  91.7362, "Guwahati"),
    "MYD"  : (25.1500,  92.7667, "Manderdisa"),
    "NHLG" : (25.0667,  92.9667, "Haflong"),
    "BPB"  : (24.8667,  92.5833, "Badarpur"),
    "NKMG" : (24.8667,  92.3500, "Karimganj"),
    "DMR"  : (24.3667,  92.1667, "Dharmanagar"),
    "AGTL" : (23.8315,  91.2868, "Agartala"),
    "NDLS" : (28.6419,  77.2194, "New Delhi"),
    "BCT"  : (18.9400,  72.8261, "Mumbai Central"),
    "MAS"  : (13.0827,  80.2707, "Chennai Central"),
    "SBC"  : (12.9784,  77.5708, "Bangalore"),
    "HWH"  : (22.5839,  88.3425, "Howrah"),
    "PUNE" : (18.5204,  73.8567, "Pune"),
    "ADI"  : (23.0225,  72.5714, "Ahmedabad"),
    "NGP"  : (21.1458,  79.0882, "Nagpur"),
    "JP"   : (26.9124,  75.7873, "Jaipur"),
    "LKO"  : (26.8467,  80.9462, "Lucknow"),
}

# ══════════════════════════════════════════════════════════════════════════════
#  APP UI
# ══════════════════════════════════════════════════════════════════════════════

APP_TITLE       = "🚂 RouteMATE AI"
APP_SUBTITLE    = "Smart Indian Railway Route Planner"
APP_ICON        = "🚂"
APP_VERSION     = "1.0.0"

# Popular route pairs shown as quick-select in UI
POPULAR_ROUTES = [
    ("KOAA", "GHY",  "Kolkata → Guwahati"),
    ("KOAA", "NDLS", "Kolkata → New Delhi"),
    ("NDLS", "BCT",  "New Delhi → Mumbai"),
    ("MAS",  "SBC",  "Chennai → Bangalore"),
    ("NDLS", "LKO",  "New Delhi → Lucknow"),
    ("HWH",  "AGTL", "Howrah → Agartala"),
]

# Streamlit page config
PAGE_CONFIG = {
    "page_title"  : APP_TITLE,
    "page_icon"   : APP_ICON,
    "layout"      : "wide",
    "initial_sidebar_state": "expanded",
}

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE   = os.path.join(ROOT_DIR, "routemate.log")

# Delay thresholds (in minutes)
DELAY_THRESHOLDS = {
    "low": 30,
    "medium": 90,
    "high": 180
}