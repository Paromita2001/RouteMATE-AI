"""
=============================================================
  RouteMATE_AI — Helper Utilities
  File: src/utils/helpers.py
=============================================================
  Shared utility functions used across all modules.
  No heavy dependencies — import freely anywhere.
=============================================================
"""

import os
import math
import logging
import datetime
import functools
import time
import pandas as pd
import numpy as np
from typing import Optional, Union

from src.utils.constants import (
    DELAY_THRESHOLDS, RISK_LOW_MAX, RISK_MEDIUM_MAX,
    RISK_LABELS, SCORE_LABELS, FEATURE_DISPLAY_NAMES,
    LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    TIME_BLOCK_MAP, DISTANCE_BAND_MAP, TRAIN_CATEGORY_CODE,
)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str) -> logging.Logger:
    """Get a consistent logger for any module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        fmt = logging.Formatter(LOG_FORMAT)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler (optional — skip if path not writable)
        try:
            fh = logging.FileHandler(LOG_FILE)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            pass

    return logger


# ══════════════════════════════════════════════════════════════════════════════
#  TIME UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def minutes_to_hhmm(minutes: float) -> str:
    """Convert float minutes → 'HH:MM' string. Returns 'N/A' on bad input."""
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "N/A"
    total = int(minutes)
    h = (total // 60) % 24
    m = total % 60
    return f"{h:02d}:{m:02d}"


def hhmm_to_minutes(time_str: str) -> Optional[float]:
    """Convert 'HH:MM' string → total minutes from midnight. Returns None on failure."""
    if not time_str or str(time_str).strip().lower() in ("source", "destination", "n/a", "nan", "none", ""):
        return None
    try:
        h, m = str(time_str).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def format_duration(minutes: float) -> str:
    """
    Format minutes into a readable duration string.
    Examples: 65 → '1h 5m', 720 → '12h 0m', 30 → '30m'
    """
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "N/A"
    total = int(minutes)
    h = total // 60
    m = total % 60
    if h == 0:
        return f"{m}m"
    return f"{h}h {m}m"


def get_time_block(minutes: float) -> str:
    """Return time block name for a given minute-of-day value."""
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "Unknown"
    h = int(minutes) // 60
    if   5  <= h < 12: return "Morning"
    elif 12 <= h < 17: return "Afternoon"
    elif 17 <= h < 21: return "Evening"
    else:              return "Night"


def get_time_block_code(minutes: float) -> int:
    return TIME_BLOCK_MAP.get(get_time_block(minutes), 4)


def travel_date_from_str(date_str: str) -> Optional[datetime.date]:
    """Parse 'YYYY-MM-DD' or 'DD-MM-YYYY' or 'DD/MM/YYYY' → date object."""
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    return None


def days_until(travel_date: datetime.date) -> int:
    """Return number of days from today until travel_date (negative if past)."""
    return (travel_date - datetime.date.today()).days


# ══════════════════════════════════════════════════════════════════════════════
#  DELAY & RISK UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def delay_category_from_minutes(minutes: float) -> str:
    """Classify a delay in minutes into On Time / Slight Delay / Significantly Delayed."""
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return "Unknown"
    if minutes <= DELAY_THRESHOLDS["on_time"]:
        return "On Time"
    if minutes <= DELAY_THRESHOLDS["slight"]:
        return "Slight Delay"
    return "Significantly Delayed"


def risk_label_from_minutes(minutes: float) -> str:
    """Return emoji risk label from predicted delay in minutes."""
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return RISK_LABELS["medium"]
    if minutes <= RISK_LOW_MAX:
        return RISK_LABELS["low"]
    if minutes <= RISK_MEDIUM_MAX:
        return RISK_LABELS["medium"]
    return RISK_LABELS["high"]


def route_score_label(score: float) -> str:
    """Return star label for a route score (0–100)."""
    for threshold, label in sorted(SCORE_LABELS.items(), reverse=True):
        if score >= threshold:
            return label
    return "⚠️ High Risk"


def compute_route_score(
    avg_delay_min   : float,
    changes         : int,
    weather_risk    : float = 1.0,
    occupancy_factor: float = 1.0,
) -> float:
    """
    Compute a 0–100 route score (higher = better / smarter choice).

    Factors:
      - Predicted delay (main driver)
      - Number of interchanges
      - Weather risk multiplier
      - Demand/occupancy factor
    """
    # Delay penalty (0–60 points)
    delay_penalty = min(avg_delay_min / 4, 60)

    # Change penalty (10 pts per change)
    change_penalty = changes * 10

    # Weather penalty (0–15 pts)
    weather_penalty = (weather_risk - 1.0) * 15

    # Demand penalty (0–10 pts)
    demand_penalty = (occupancy_factor - 1.0) * 10

    total_penalty = delay_penalty + change_penalty + weather_penalty + demand_penalty
    score = max(0.0, min(100.0 - total_penalty, 100.0))
    return round(score, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  STATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def standardize_station_code(code: str) -> str:
    """Uppercase, strip whitespace from station code."""
    if not code:
        return ""
    return str(code).strip().upper()


def parse_station_list(raw) -> list:
    """
    Safely parse a stationList field (JSON string or Python literal).
    Returns empty list on failure.
    """
    import json, ast
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(str(raw))
        except Exception:
            return []


def running_days_count(running_days_str: str) -> int:
    """Count number of days a train runs per week from 'MON,WED,FRI' string."""
    if not running_days_str or pd.isna(running_days_str):
        return 7
    return len(str(running_days_str).strip().split(","))


def is_daily_train(running_days_str: str) -> bool:
    return running_days_count(running_days_str) == 7


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_distance_band(km: float) -> str:
    if km <= 200:   return "Short"
    if km <= 800:   return "Medium"
    return "Long"


def get_distance_band_code(km: float) -> int:
    return DISTANCE_BAND_MAP.get(get_distance_band(km), 1)


def get_train_cat_code(category: str) -> int:
    return TRAIN_CATEGORY_CODE.get(category, 0)


def build_feature_row(
    station_code    : str,
    train_category  : str,
    stop_position   : float,
    distance_km     : float,
    halt_min        : float,
    time_of_day_min : float,
    is_overnight    : int,
    journey_day     : int,
    running_days    : str,
    total_stops     : int,
    station_pct_rt  : float,
    station_pct_sl  : float,
    station_pct_sig : float,
) -> dict:
    """
    Build a single feature dict ready for model.predict().
    Used by predict.py and the Streamlit app.
    """
    return {
        "train_cat_code"        : get_train_cat_code(train_category),
        "stop_position_ratio"   : stop_position,
        "distance_km"           : distance_km,
        "halt_min"              : min(halt_min, 120),
        "time_of_day_min"       : time_of_day_min,
        "time_block_code"       : get_time_block_code(time_of_day_min),
        "is_overnight"          : int(is_overnight),
        "journey_day"           : int(journey_day),
        "running_days_count"    : running_days_count(running_days),
        "is_daily"              : int(is_daily_train(running_days)),
        "total_stops"           : total_stops,
        "station_pct_right_time": station_pct_rt,
        "station_pct_slight"    : station_pct_sl,
        "station_pct_sig"       : station_pct_sig,
        "distance_band_code"    : get_distance_band_code(distance_km),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FARE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def format_inr(amount: float) -> str:
    """Format a number as Indian Rupees with commas. e.g. 12345 → '₹12,345'"""
    try:
        return f"₹{int(amount):,}"
    except Exception:
        return f"₹{amount}"


def fare_value_label(cost_per_km: float) -> str:
    """Return value rating label based on cost per km."""
    if cost_per_km < 0.8:   return "⭐⭐⭐ Best Value"
    if cost_per_km < 2.0:   return "⭐⭐ Good Value"
    if cost_per_km < 3.5:   return "⭐ Fair"
    return "💸 Premium"


# ══════════════════════════════════════════════════════════════════════════════
#  DATA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_station_code(code: str, valid_codes: set) -> tuple[bool, str]:
    """
    Validate that a station code exists in the graph.
    Returns (is_valid, error_message).
    """
    code = standardize_station_code(code)
    if not code:
        return False, "Station code cannot be empty."
    if code not in valid_codes:
        return False, f"Station '{code}' not found. Check the code and try again."
    return True, ""


def validate_travel_date(date: datetime.date) -> tuple[bool, str]:
    """Check that travel date is today or in the future."""
    today = datetime.date.today()
    if date < today:
        return False, f"Travel date {date} is in the past."
    if date > today + datetime.timedelta(days=120):
        return False, "Cannot search routes more than 120 days in advance."
    return True, ""


def validate_passenger_count(n: int) -> tuple[bool, str]:
    if n < 1:
        return False, "At least 1 passenger required."
    if n > 6:
        return False, "Maximum 6 passengers per booking."
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY / FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def feature_display_name(col: str) -> str:
    """Return human-readable name for a feature column."""
    return FEATURE_DISPLAY_NAMES.get(col, col.replace("_", " ").title())


def summarize_route(route: dict) -> str:
    """One-line summary of a route for display in lists."""
    rtype  = route.get("route_type", "Route")
    train  = route.get("train_name", "Multi-train")
    hrs    = route.get("total_travel_hrs") or (route.get("total_travel_min", 0) / 60)
    score  = route.get("route_score", "—")
    risk   = route.get("overall_risk", "")
    return f"{rtype} | {train} | {hrs:.1f} hrs | Score: {score} | {risk}"


def format_legs_text(route: dict) -> str:
    """Format all legs of a route into readable multi-line text."""
    lines = []
    for leg in route.get("legs", []):
        fr   = leg.get("from_name", leg.get("from", "?"))
        to   = leg.get("to_name",   leg.get("to",   "?"))
        trn  = leg.get("train_number", "?")
        dep  = leg.get("depart_time", "?")
        arr  = leg.get("arrive_time", "?")
        mins = leg.get("travel_min")
        dur  = format_duration(mins) if mins else "?"
        lines.append(f"  🚂 {fr} → {to}  [Train {trn}]  {dep} → {arr}  ({dur})")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE / CACHING
# ══════════════════════════════════════════════════════════════════════════════

def timed(func):
    """Decorator — logs execution time of any function."""
    logger = get_logger("timer")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns default instead of raising ZeroDivisionError."""
    try:
        return a / b if b != 0 else default
    except Exception:
        return default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(value, max_val))


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Helpers — Test")
    print("=" * 55)

    # Time utils
    print(f"\n  minutes_to_hhmm(585)       = {minutes_to_hhmm(585)}")
    print(f"  hhmm_to_minutes('09:45')   = {hhmm_to_minutes('09:45')}")
    print(f"  format_duration(1085)      = {format_duration(1085)}")
    print(f"  get_time_block(585)        = {get_time_block(585)}")

    # Delay utils
    print(f"\n  delay_category(7)          = {delay_category_from_minutes(7)}")
    print(f"  delay_category(45)         = {delay_category_from_minutes(45)}")
    print(f"  delay_category(230)        = {delay_category_from_minutes(230)}")
    print(f"  risk_label(7)              = {risk_label_from_minutes(7)}")
    print(f"  risk_label(230)            = {risk_label_from_minutes(230)}")

    # Route score
    score = compute_route_score(avg_delay_min=20, changes=0, weather_risk=1.0, occupancy_factor=1.0)
    print(f"\n  route_score(delay=20, changes=0) = {score}  → {route_score_label(score)}")
    score2 = compute_route_score(avg_delay_min=150, changes=1, weather_risk=1.4, occupancy_factor=1.7)
    print(f"  route_score(delay=150, changes=1, weather, peak) = {score2}  → {route_score_label(score2)}")

    # Feature helpers
    print(f"\n  get_distance_band(50)      = {get_distance_band(50)}")
    print(f"  get_distance_band(500)     = {get_distance_band(500)}")
    print(f"  get_distance_band(1200)    = {get_distance_band(1200)}")
    print(f"  running_days_count('MON,WED,FRI') = {running_days_count('MON,WED,FRI')}")
    print(f"  is_daily_train('SUN,MON,TUE,WED,THU,FRI,SAT') = {is_daily_train('SUN,MON,TUE,WED,THU,FRI,SAT')}")

    # Fare
    print(f"\n  format_inr(12345)          = {format_inr(12345)}")
    print(f"  fare_value_label(0.62)     = {fare_value_label(0.62)}")
    print(f"  fare_value_label(2.5)      = {fare_value_label(2.5)}")

    # Validation
    ok, msg = validate_station_code("KOAA", {"KOAA", "GHY", "NJP"})
    print(f"\n  validate_station_code('KOAA') = ({ok}, '{msg}')")
    ok2, msg2 = validate_station_code("XYZ", {"KOAA", "GHY"})
    print(f"  validate_station_code('XYZ')  = ({ok2}, '{msg2}')")

    # Display
    print(f"\n  feature_display_name('station_pct_sig') = {feature_display_name('station_pct_sig')}")
    print(f"  feature_display_name('distance_km')     = {feature_display_name('distance_km')}")

def format_inr(amount):
    if amount is None:
        return "₹0"
    return f"₹{int(amount):,}"