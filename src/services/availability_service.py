"""
=============================================================
  RouteMATE_AI — Availability Service
  File: src/services/availability_service.py
=============================================================
  Estimates seat availability for trains based on:
    - Train type (Rajdhani / Superfast / Express / Passenger)
    - Travel date demand (from calendar_service)
    - Distance / journey length
    - Class of travel
    - Time until departure (advance booking effect)

  No paid API needed — uses Indian Railways capacity
  standards + demand modelling.

  Optional: plug in RapidAPI / IRCTC API key for live data.

  Key outputs:
    - availability_status  (Available / Limited / Waitlist / RAC)
    - estimated_seats      (integer estimate)
    - recommended_class    (best value class)
    - booking_urgency      (Book Now / Book Soon / Flexible)
    - waitlist_estimate    (WL position estimate)
=============================================================
"""

import datetime
import math
from typing import Optional

# ── Indian Railways standard coach capacity by class ─────────────────────────
COACH_CAPACITY = {
    "1A" : 24,    # First AC (4 berth coupe)
    "2A" : 46,    # Second AC (2 tier)
    "3A" : 64,    # Third AC (3 tier)
    "SL" : 72,    # Sleeper
    "CC" : 78,    # AC Chair Car
    "2S" : 108,   # Second Sitting
    "GEN": 90,    # General (unreserved)
}

# ── Standard number of coaches per train type per class ──────────────────────
TRAIN_COACH_CONFIG = {
    "Rajdhani" : {"1A": 1, "2A": 4, "3A": 6,  "CC": 0, "SL": 0,  "2S": 0},
    "Humsafar" : {"1A": 0, "2A": 0, "3A": 20, "CC": 0, "SL": 0,  "2S": 0},
    "Tejas"    : {"1A": 0, "2A": 0, "3A": 0,  "CC": 12,"SL": 0,  "2S": 0},
    "Superfast" : {"1A": 1, "2A": 2, "3A": 5,  "CC": 2, "SL": 8,  "2S": 2},
    "Mail/Express":{"1A":0, "2A": 1, "3A": 3,  "CC": 0, "SL": 10, "2S": 3},
    "Express"  : {"1A": 0, "2A": 1, "3A": 3,  "CC": 0, "SL": 10, "2S": 3},
    "Passenger" : {"1A": 0, "2A": 0, "3A": 0,  "CC": 0, "SL": 2,  "2S": 8},
}

# ── Typical fare per km per class (₹) ─────────────────────────────────────────
FARE_PER_KM = {
    "1A" : 4.50,
    "2A" : 2.80,
    "3A" : 2.00,
    "SL" : 0.90,
    "CC" : 1.60,
    "2S" : 0.50,
    "GEN": 0.35,
}

BASE_CHARGE = {
    "1A": 300, "2A": 200, "3A": 150,
    "SL": 50,  "CC": 100, "2S": 30, "GEN": 20,
}

# Reservation charge
RESERVATION_CHARGE = {
    "1A": 60, "2A": 50, "3A": 40,
    "SL": 25, "CC": 40, "2S": 15, "GEN": 0,
}


def _total_capacity(train_type: str, travel_class: str) -> int:
    config = TRAIN_COACH_CONFIG.get(train_type, TRAIN_COACH_CONFIG["Express"])
    coaches = config.get(travel_class, 0)
    return coaches * COACH_CAPACITY.get(travel_class, 72)


def _demand_factor(
    occupancy_factor: float,   # from calendar_service
    days_to_travel: int,
    distance_km: float,
) -> float:
    """
    Compute how full the train is likely to be (0.0 → 1.0).
    """
    # Base demand by days to travel
    if days_to_travel <= 0:
        base = 0.97
    elif days_to_travel <= 3:
        base = 0.85
    elif days_to_travel <= 7:
        base = 0.72
    elif days_to_travel <= 15:
        base = 0.58
    elif days_to_travel <= 30:
        base = 0.45
    else:
        base = 0.32

    # Long distance journeys fill up faster
    dist_factor = 1.1 if distance_km > 1000 else 1.05 if distance_km > 500 else 1.0

    # Occupancy factor adds pressure (peak season boost, not full multiplier)
    occ_boost = (occupancy_factor - 1.0) * 0.20

    return min(base * dist_factor + occ_boost, 1.0)


def _status_from_fill(fill_rate: float, capacity: int) -> tuple:
    """
    Convert fill rate → (status, estimated_seats, wl_estimate).
    """
    available = max(0, int(capacity * (1 - fill_rate)))

    if fill_rate < 0.70:
        status  = "Available"
        wl      = 0
    elif fill_rate < 0.85:
        status  = "Limited"
        wl      = 0
    elif fill_rate < 0.95:
        status  = "RAC"          # Reservation Against Cancellation
        rac_count = int(capacity * 0.05)
        available = rac_count
        wl        = 0
    else:
        status    = "Waitlist"
        available = 0
        wl        = max(1, int((fill_rate - 0.95) * capacity * 3))

    return status, available, wl


def _booking_urgency(status: str, days_to_travel: int, fill_rate: float) -> str:
    if status == "Waitlist":
        return "⚠️ Book Tatkal or choose alternate train"
    if status == "RAC":
        return "🔶 Book immediately — RAC only"
    if status == "Limited" or (fill_rate > 0.6 and days_to_travel < 7):
        return "🟡 Book Soon"
    if days_to_travel > 30:
        return "🟢 Flexible — plenty of time"
    return "🟢 Book Now (good availability)"


def _recommended_class(train_type: str, distance_km: float, budget: str = "medium") -> str:
    """Suggest best class for journey type and budget."""
    if train_type in ("Rajdhani", "Humsafar"):
        return "3A" if budget != "premium" else "2A"
    if distance_km > 800:
        if budget == "budget":    return "SL"
        if budget == "medium":    return "3A"
        return "2A"
    if distance_km > 300:
        if budget == "budget":    return "SL"
        return "3A"
    return "2S" if budget == "budget" else "CC"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AVAILABILITY FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def get_availability(
    train_number   : int,
    train_type     : str,
    travel_class   : str,
    travel_date    : datetime.date,
    distance_km    : float,
    occupancy_factor: float = 1.0,    # from calendar_service
) -> dict:
    """
    Estimate seat availability for a specific train + class + date.

    Args:
        train_number    : e.g. 12525
        train_type      : Rajdhani / Superfast / Mail/Express / Passenger
        travel_class    : 1A / 2A / 3A / SL / CC / 2S
        travel_date     : date of journey
        distance_km     : total journey distance
        occupancy_factor: from calendar_service.get_calendar_info()

    Returns dict with status, seats, urgency, recommendation.
    """
    today           = datetime.date.today()
    days_to_travel  = (travel_date - today).days
    capacity        = _total_capacity(train_type, travel_class)

    if capacity == 0:
        return {
            "train_number"      : train_number,
            "travel_class"      : travel_class,
            "travel_date"       : str(travel_date),
            "availability_status": "Not Available",
            "estimated_seats"   : 0,
            "waitlist_estimate" : 0,
            "fill_rate_pct"     : 100,
            "booking_urgency"   : "❌ This class not available on this train type",
            "recommended_class" : _recommended_class(train_type, distance_km),
            "note"              : f"{travel_class} class not present on {train_type} trains",
        }

    fill_rate = _demand_factor(occupancy_factor, days_to_travel, distance_km)
    status, seats, wl = _status_from_fill(fill_rate, capacity)
    urgency = _booking_urgency(status, days_to_travel, fill_rate)
    rec_class = _recommended_class(train_type, distance_km)

    return {
        "train_number"        : train_number,
        "train_type"          : train_type,
        "travel_class"        : travel_class,
        "travel_date"         : str(travel_date),
        "days_to_travel"      : days_to_travel,
        "availability_status" : status,
        "estimated_seats"     : seats,
        "total_capacity"      : capacity,
        "fill_rate_pct"       : round(fill_rate * 100, 1),
        "waitlist_estimate"   : wl,
        "booking_urgency"     : urgency,
        "recommended_class"   : rec_class,
        "note"                : (
            f"WL ~{wl}" if status == "Waitlist"
            else f"~{seats} seats available" if seats > 0
            else "RAC available"
        ),
    }


def get_all_class_availability(
    train_number    : int,
    train_type      : str,
    travel_date     : datetime.date,
    distance_km     : float,
    occupancy_factor: float = 1.0,
) -> dict:
    """Check availability across all classes for a train."""
    results = {}
    for cls in ["1A", "2A", "3A", "SL", "CC", "2S"]:
        a = get_availability(
            train_number, train_type, cls,
            travel_date, distance_km, occupancy_factor
        )
        if a["availability_status"] != "Not Available":
            results[cls] = a

    # Best available class
    priority = ["3A", "SL", "2A", "CC", "2S", "1A"]
    best = next(
        (c for c in priority if results.get(c, {}).get("availability_status") == "Available"),
        None
    )

    return {
        "train_number"   : train_number,
        "travel_date"    : str(travel_date),
        "by_class"       : results,
        "best_class"     : best,
        "summary"        : (
            f"Best option: {best}" if best
            else "Limited / Waitlist only — consider Tatkal"
        ),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import datetime

    print("=" * 55)
    print("  AvailabilityService — Test")
    print("=" * 55)

    travel_date = datetime.date.today() + datetime.timedelta(days=5)

    # Test single class
    result = get_availability(
        train_number=13181,
        train_type="Mail/Express",
        travel_class="SL",
        travel_date=travel_date,
        distance_km=1144,
        occupancy_factor=1.5,   # peak season
    )
    print(f"\n  Train 13181 | SL | {travel_date}")
    for k, v in result.items():
        print(f"    {k:<25}: {v}")

    # Test all classes
    print(f"\n  All classes for Train 12525 (Superfast):")
    all_cls = get_all_class_availability(
        train_number=12525,
        train_type="Superfast",
        travel_date=travel_date,
        distance_km=1000,
        occupancy_factor=1.3,
    )
    print(f"    Best class : {all_cls['best_class']}")
    print(f"    Summary    : {all_cls['summary']}")
    for cls, info in all_cls["by_class"].items():
        print(f"    {cls}: {info['availability_status']:<12} {info['booking_urgency']}")