"""
=============================================================
  RouteMATE_AI — Calendar Service
  File: src/services/calendar_service.py
=============================================================
  Detects Indian public holidays, festival seasons,
  school vacations, and peak travel periods that
  significantly affect train availability and delays.

  No external API needed — fully offline using a
  curated Indian railway calendar model.

  Key outputs:
    - is_holiday         (bool)
    - is_peak_season     (bool)
    - holiday_name       (str)
    - occupancy_factor   (1.0 = normal, 2.0 = double demand)
    - delay_risk_boost   (extra minutes to add to delay estimate)
    - travel_advisory    (plain text)
=============================================================
"""

import datetime
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  FIXED INDIAN PUBLIC HOLIDAYS (day, month)
# ══════════════════════════════════════════════════════════════════════════════

FIXED_HOLIDAYS = {
    (1,  1):  "New Year's Day",
    (26, 1):  "Republic Day",
    (14, 4):  "Dr. Ambedkar Jayanti / Tamil New Year",
    (1,  5):  "Labour Day / May Day",
    (15, 8):  "Independence Day",
    (2,  10): "Gandhi Jayanti",
    (25, 12): "Christmas Day",
}

# ── Festival seasons (approximate ranges, recalculated yearly) ────────────────
# These shift each year but the ranges below are representative
FESTIVAL_SEASONS = [
    # (month, start_day, end_day, name, occupancy_factor, delay_boost)
    (1,  13, 17,  "Makar Sankranti / Lohri",   1.4,  15),
    (2,  25, 28,  "Holi Season",                1.6,  20),
    (3,  22, 29,  "Holi Peak Travel",            1.8,  30),
    (4,  10, 17,  "Baisakhi / Easter",           1.3,  10),
    (6,  1,  15,  "Summer Vacation Peak",        1.9,  35),
    (7,  1,  15,  "Summer Vacation (Late)",      1.7,  25),
    (8,  12, 20,  "Independence Day Weekend",    1.4,  15),
    (9,  1,  15,  "Ganesh Chaturthi Season",     1.5,  20),
    (10, 1,  26,  "Navratri / Dussehra",         1.7,  25),
    (10, 20, 31,  "Diwali Approach",             1.9,  35),
    (11, 1,  5,   "Diwali Peak",                 2.0,  40),
    (11, 6,  15,  "Post-Diwali Return",          1.8,  30),
    (12, 20, 31,  "Christmas / New Year Rush",   1.8,  30),
    (12, 1,  10,  "Winter Vacation Start",       1.5,  20),
]

# ── School vacation periods ────────────────────────────────────────────────────
SCHOOL_VACATIONS = [
    (5,  15, 6,  15,  "Summer School Vacation"),
    (10, 15, 11, 5,   "Dussehra / Diwali School Break"),
    (12, 20, 1,  5,   "Winter School Break"),
]

# ── Weekend type ──────────────────────────────────────────────────────────────
# In India, Friday evening + Saturday + Sunday = high travel
WEEKEND_BOOST = {
    4: (1.3, 10),   # Friday (weekday=4)
    5: (1.5, 15),   # Saturday
    6: (1.4, 12),   # Sunday
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CALENDAR ANALYSER
# ══════════════════════════════════════════════════════════════════════════════

def get_calendar_info(date: datetime.date = None) -> dict:
    """
    Analyse a travel date and return occupancy/delay risk context.

    Returns:
        is_holiday        : bool
        is_peak_season    : bool
        is_weekend        : bool
        is_school_vacation: bool
        holiday_name      : str or None
        festival_name     : str or None
        occupancy_factor  : float  (1.0 normal → 2.0 fully packed)
        delay_risk_boost  : int    (extra minutes of expected delay)
        travel_advisory   : str
        risk_label        : str  (🟢/🟡/🔴)
    """
    if date is None:
        date = datetime.date.today()

    day   = date.day
    month = date.month
    wday  = date.weekday()   # 0=Mon … 6=Sun

    result = {
        "date"              : str(date),
        "day_of_week"       : date.strftime("%A"),
        "is_holiday"        : False,
        "is_peak_season"    : False,
        "is_weekend"        : wday >= 5,
        "is_school_vacation": False,
        "holiday_name"      : None,
        "festival_name"     : None,
        "occupancy_factor"  : 1.0,
        "delay_risk_boost"  : 0,
        "travel_advisory"   : "",
    }

    advisories = []

    # ── Check fixed holidays ──────────────────────────────────────────────────
    h_name = FIXED_HOLIDAYS.get((day, month))
    if h_name:
        result["is_holiday"]       = True
        result["holiday_name"]     = h_name
        result["occupancy_factor"] = max(result["occupancy_factor"], 1.6)
        result["delay_risk_boost"] += 20
        advisories.append(f"🎉 Public holiday: {h_name}")

    # ── Check festival seasons ────────────────────────────────────────────────
    for (m, sd, ed, name, occ, db) in FESTIVAL_SEASONS:
        if month == m and sd <= day <= ed:
            result["is_peak_season"]   = True
            result["festival_name"]    = name
            result["occupancy_factor"] = max(result["occupancy_factor"], occ)
            result["delay_risk_boost"] = max(result["delay_risk_boost"], db)
            advisories.append(f"🪔 Festival season: {name} — high occupancy expected")
            break

    # ── Check school vacations ────────────────────────────────────────────────
    for (sm, sday, em, eday, vname) in SCHOOL_VACATIONS:
        in_vacation = False
        if sm == em:
            if month == sm and sday <= day <= eday:
                in_vacation = True
        else:
            if (month == sm and day >= sday) or (month == em and day <= eday):
                in_vacation = True
            elif sm < month < em:
                in_vacation = True

        if in_vacation:
            result["is_school_vacation"] = True
            result["occupancy_factor"]   = max(result["occupancy_factor"], 1.5)
            result["delay_risk_boost"]  += 15
            advisories.append(f"🏫 School vacation: {vname} — families travelling")
            break

    # ── Weekend effect ────────────────────────────────────────────────────────
    if wday in WEEKEND_BOOST:
        wk_occ, wk_db = WEEKEND_BOOST[wday]
        result["occupancy_factor"] = max(result["occupancy_factor"], wk_occ)
        result["delay_risk_boost"] = max(result["delay_risk_boost"], wk_db)
        advisories.append(f"📅 {result['day_of_week']} — weekend travel demand")

    # ── Monsoon season boost (June-September) ────────────────────────────────
    if month in (6, 7, 8, 9):
        result["delay_risk_boost"] += 10
        advisories.append("🌧️ Monsoon season — weather-related delays possible")

    # ── Compose advisory ─────────────────────────────────────────────────────
    if not advisories:
        advisories.append("✅ Normal travel day — no special conditions")

    result["travel_advisory"] = " | ".join(advisories)

    # ── Risk label ────────────────────────────────────────────────────────────
    occ = result["occupancy_factor"]
    if occ >= 1.8:
        result["risk_label"] = "🔴 Very High Demand"
    elif occ >= 1.4:
        result["risk_label"] = "🟡 High Demand"
    else:
        result["risk_label"] = "🟢 Normal"

    return result


def get_best_travel_days(from_date: datetime.date, days_ahead: int = 7) -> list:
    """
    Return ranked list of travel days in the next N days,
    best (lowest occupancy) first.
    """
    options = []
    for i in range(days_ahead):
        d = from_date + datetime.timedelta(days=i)
        info = get_calendar_info(d)
        options.append({
            "date"             : d,
            "date_str"         : d.strftime("%d %b %Y (%A)"),
            "occupancy_factor" : info["occupancy_factor"],
            "delay_risk_boost" : info["delay_risk_boost"],
            "risk_label"       : info["risk_label"],
            "note"             : info["festival_name"] or info["holiday_name"] or "Normal",
        })

    options.sort(key=lambda x: (x["occupancy_factor"], x["delay_risk_boost"]))
    return options


def is_tatkal_recommended(date: datetime.date) -> tuple[bool, str]:
    """
    Returns (True, reason) if Tatkal booking is recommended
    due to high demand on the given date.
    """
    info = get_calendar_info(date)
    if info["occupancy_factor"] >= 1.7:
        reason = (
            f"High demand expected ({info['festival_name'] or info['holiday_name'] or 'Peak season'}). "
            f"Book Tatkal or confirm tickets well in advance."
        )
        return True, reason
    return False, "Regular booking should be fine for this date."


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  CalendarService — Test")
    print("=" * 55)

    test_dates = [
        datetime.date.today(),
        datetime.date(2025, 10, 28),   # Diwali zone
        datetime.date(2025, 1, 26),    # Republic Day
        datetime.date(2025, 6, 10),    # Summer vacation peak
        datetime.date(2025, 12, 25),   # Christmas
    ]

    for d in test_dates:
        info = get_calendar_info(d)
        print(f"\n  📅 {d.strftime('%d %b %Y (%A)')}")
        print(f"     Holiday      : {info['holiday_name'] or 'None'}")
        print(f"     Festival     : {info['festival_name'] or 'None'}")
        print(f"     Occupancy ×  : {info['occupancy_factor']}")
        print(f"     Delay boost  : +{info['delay_risk_boost']} min")
        print(f"     Risk         : {info['risk_label']}")
        print(f"     Advisory     : {info['travel_advisory']}")

    print("\n\n  📆 Best 7 days to travel from today:")
    best = get_best_travel_days(datetime.date.today(), days_ahead=7)
    for b in best:
        print(f"     {b['date_str']:<30} {b['risk_label']:<22} Note: {b['note']}")

    rec, reason = is_tatkal_recommended(datetime.date(2025, 11, 2))
    print(f"\n  Tatkal recommended for 2 Nov 2025? {rec} — {reason}")

