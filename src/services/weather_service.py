"""
=============================================================
  RouteMATE_AI — Weather Service
  File: src/services/weather_service.py
=============================================================
  Fetches real-time weather for any station city using
  Open-Meteo API (100% free, no API key needed).

  Falls back to a seasonal heuristic model when offline.

  Key outputs per station:
    - temperature, humidity, wind speed
    - weather condition (Clear / Rain / Fog / Storm / Snow)
    - delay_risk_factor  (multiplier: 1.0 = normal, 2.5 = severe)
    - weather_advisory   (plain text warning)
=============================================================
"""

import math
import datetime
import requests

# ── Station → approximate lat/lon ────────────────────────────────────────────
# Covers all 21 stations in your delay dataset + common route stations
STATION_COORDINATES = {
    "KOAA" : (22.5726, 88.3639, "Kolkata"),
    "BDC"  : (23.0333, 88.3833, "Bandel"),
    "NDAE" : (23.4000, 88.3667, "Nabadwip"),
    "KWAE" : (23.6500, 88.1333, "Katwa"),
    "AZ"   : (24.0667, 88.2667, "Azimganj"),
    "JRLE" : (24.4667, 88.0667, "Jangipur"),
    "MLDT" : (25.0000, 88.1333, "Malda"),
    "KNE"  : (26.1000, 87.9500, "Kishanganj"),
    "NJP"  : (26.7100, 88.3547, "New Jalpaiguri"),
    "NCB"  : (26.3333, 89.4667, "New Cooch Behar"),
    "NOQ"  : (26.5000, 89.8333, "New Alipurduar"),
    "KOJ"  : (26.3833, 90.2667, "Kokrajhar"),
    "NBQ"  : (26.4833, 90.5583, "New Bongaigaon"),
    "GLPT" : (26.1833, 90.6333, "Goalpara"),
    "GHY"  : (26.1445, 91.7362, "Guwahati"),
    "MYD"  : (25.1500, 92.7667, "Manderdisa"),
    "NHLG" : (25.0667, 92.9667, "Haflong"),
    "BPB"  : (24.8667, 92.5833, "Badarpur"),
    "NKMG" : (24.8667, 92.3500, "Karimganj"),
    "DMR"  : (24.3667, 92.1667, "Dharmanagar"),
    "AGTL" : (23.8315, 91.2868, "Agartala"),
    # Additional major stations
    "NDLS" : (28.6419, 77.2194, "New Delhi"),
    "BCT"  : (18.9400, 72.8261, "Mumbai"),
    "MAS"  : (13.0827, 80.2707, "Chennai"),
    "SBC"  : (12.9784, 77.5708, "Bangalore"),
    "HWH"  : (22.5839, 88.3425, "Howrah"),
    "PUNE" : (18.5204, 73.8567, "Pune"),
    "ADI"  : (23.0225, 72.5714, "Ahmedabad"),
    "NGP"  : (21.1458, 79.0882, "Nagpur"),
    "JP"   : (26.9124, 75.7873, "Jaipur"),
    "LKO"  : (26.8467, 80.9462, "Lucknow"),
}


# ── Weather condition → delay risk multiplier ─────────────────────────────────
WEATHER_RISK = {
    "Clear"        : 1.0,
    "Partly Cloudy": 1.0,
    "Cloudy"       : 1.1,
    "Fog"          : 1.8,   # serious delay risk on Indian railways
    "Drizzle"      : 1.2,
    "Rain"         : 1.4,
    "Heavy Rain"   : 1.7,
    "Thunderstorm" : 2.0,
    "Snow"         : 2.5,
    "Haze"         : 1.3,
}

# Open-Meteo WMO weather code → condition string
def _wmo_to_condition(code: int) -> str:
    if code == 0:                    return "Clear"
    if code in (1, 2):               return "Partly Cloudy"
    if code == 3:                    return "Cloudy"
    if code in (45, 48):             return "Fog"
    if code in (51, 53, 55):         return "Drizzle"
    if code in (61, 63):             return "Rain"
    if code in (65, 66, 67):         return "Heavy Rain"
    if code in (71, 73, 75, 77):     return "Snow"
    if code in (80, 81, 82):         return "Rain"
    if code in (95, 96, 99):         return "Thunderstorm"
    return "Cloudy"


def _weather_advisory(condition: str, wind_kmh: float, temp: float) -> str:
    advisories = []
    if condition == "Fog":
        advisories.append("⚠️ Dense fog — expect signal delays and reduced visibility")
    if condition in ("Heavy Rain", "Thunderstorm"):
        advisories.append("⛈️ Severe weather — possible track flooding and cancellations")
    if condition == "Snow":
        advisories.append("❄️ Snowfall — significant delays likely on mountain routes")
    if wind_kmh > 60:
        advisories.append(f"💨 Strong winds ({wind_kmh:.0f} km/h) — may affect operations")
    if temp > 44:
        advisories.append("🌡️ Extreme heat — track expansion checks may cause delays")
    if not advisories:
        advisories.append("✅ Weather conditions normal for travel")
    return " | ".join(advisories)


# ── Main fetcher ──────────────────────────────────────────────────────────────

def get_weather(station_code: str, date: datetime.date = None) -> dict:
    """
    Fetch weather for a station. Returns a dict with condition,
    temperature, humidity, wind, delay_risk_factor, advisory.

    Uses Open-Meteo (free, no API key).
    Falls back to seasonal heuristic if request fails.
    """
    if date is None:
        date = datetime.date.today()

    code = station_code.upper()
    coords = STATION_COORDINATES.get(code)

    if coords is None:
        return _fallback_weather(code, date, reason="Station not in coordinate database")

    lat, lon, city = coords

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=weathercode,temperature_2m_max,temperature_2m_min,"
            f"precipitation_sum,windspeed_10m_max,relativehumidity_2m_max"
            f"&timezone=Asia%2FKolkata"
            f"&start_date={date}&end_date={date}"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()["daily"]

        wmo_code   = data["weathercode"][0]
        temp_max   = data["temperature_2m_max"][0]
        temp_min   = data["temperature_2m_min"][0]
        temp_avg   = round((temp_max + temp_min) / 2, 1)
        precip     = data["precipitation_sum"][0] or 0.0
        wind_kmh   = data["windspeed_10m_max"][0] or 0.0
        humidity   = data["relativehumidity_2m_max"][0] or 60

        condition  = _wmo_to_condition(wmo_code)
        risk       = WEATHER_RISK.get(condition, 1.0)
        advisory   = _weather_advisory(condition, wind_kmh, temp_max)

        return {
            "station_code"       : code,
            "city"               : city,
            "date"               : str(date),
            "condition"          : condition,
            "temperature_c"      : temp_avg,
            "temp_max_c"         : temp_max,
            "temp_min_c"         : temp_min,
            "precipitation_mm"   : precip,
            "wind_kmh"           : wind_kmh,
            "humidity_pct"       : humidity,
            "delay_risk_factor"  : risk,
            "weather_advisory"   : advisory,
            "source"             : "Open-Meteo (live)",
        }

    except Exception as e:
        return _fallback_weather(code, date, reason=str(e), coords=coords)


def _fallback_weather(code: str, date: datetime.date, reason: str = "", coords=None) -> dict:
    """Seasonal heuristic when API is unavailable."""
    month = date.month
    city  = coords[2] if coords else code

    # Northeast India (high delay stations) → monsoon heavy
    northeast = {"GHY", "NBQ", "KOJ", "NOQ", "NCB", "NJP", "AGTL", "BPB", "NKMG", "DMR", "MYD", "NHLG"}

    if month in (6, 7, 8, 9):       # Monsoon
        condition = "Heavy Rain" if code in northeast else "Rain"
        temp, humidity, wind = 28, 88, 25
    elif month in (12, 1, 2):        # Winter
        condition = "Fog" if code in {"KOAA", "MLDT", "KNE", "NJP"} else "Cloudy"
        temp, humidity, wind = 14, 75, 12
    elif month in (3, 4, 5):         # Summer
        condition = "Clear"
        temp, humidity, wind = 38, 40, 18
    else:                            # Oct-Nov post-monsoon
        condition = "Partly Cloudy"
        temp, humidity, wind = 26, 65, 15

    risk     = WEATHER_RISK.get(condition, 1.0)
    advisory = _weather_advisory(condition, wind, temp)

    return {
        "station_code"      : code,
        "city"              : city,
        "date"              : str(date),
        "condition"         : condition,
        "temperature_c"     : temp,
        "temp_max_c"        : temp + 3,
        "temp_min_c"        : temp - 3,
        "precipitation_mm"  : 15 if "Rain" in condition else 0,
        "wind_kmh"          : wind,
        "humidity_pct"      : humidity,
        "delay_risk_factor" : risk,
        "weather_advisory"  : advisory,
        "source"            : f"Seasonal heuristic (API unavailable: {reason[:60]})",
    }


def get_route_weather(station_codes: list, date: datetime.date = None) -> dict:
    """
    Fetch weather for all stations on a route.
    Returns per-station results + worst-case risk factor.
    """
    if date is None:
        date = datetime.date.today()

    results     = {}
    max_risk    = 1.0
    worst_cond  = "Clear"
    advisories  = []

    for code in station_codes:
        w = get_weather(code, date)
        results[code] = w
        if w["delay_risk_factor"] > max_risk:
            max_risk   = w["delay_risk_factor"]
            worst_cond = w["condition"]
        if w["delay_risk_factor"] > 1.2:
            advisories.append(f"{code}: {w['weather_advisory']}")

    return {
        "stations"              : results,
        "worst_condition"       : worst_cond,
        "max_delay_risk_factor" : max_risk,
        "route_advisories"      : advisories,
        "date"                  : str(date),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import datetime

    print("=" * 55)
    print("  WeatherService — Test")
    print("=" * 55)

    today = datetime.date.today()
    for code in ["KOAA", "GHY", "NJP", "AGTL"]:
        w = get_weather(code, today)
        print(f"\n  {code} ({w['city']}) — {w['date']}")
        print(f"    Condition : {w['condition']}")
        print(f"    Temp      : {w['temperature_c']}°C")
        print(f"    Wind      : {w['wind_kmh']} km/h")
        print(f"    Humidity  : {w['humidity_pct']}%")
        print(f"    Risk ×    : {w['delay_risk_factor']}")
        print(f"    Advisory  : {w['weather_advisory']}")
        print(f"    Source    : {w['source']}")

    print("\n  Route weather (KOAA → GHY route):")
    rw = get_route_weather(["KOAA", "NJP", "NBQ", "GHY"], today)
    print(f"    Worst condition    : {rw['worst_condition']}")
    print(f"    Max risk factor    : {rw['max_delay_risk_factor']}")
    print(f"    Route advisories   : {rw['route_advisories']}")