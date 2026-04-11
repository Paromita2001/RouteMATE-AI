"""
=============================================================
  RouteMATE_AI — Cost Service
  File: src/services/cost_service.py
=============================================================
  Calculates accurate Indian Railways fare estimates for
  any train type, class, and distance using the official
  IR fare structure (as of 2024).

  Covers:
    - Base fare by distance slab (IR official slabs)
    - Class multipliers (SL / 3A / 2A / 1A / CC / 2S)
    - Superfast surcharge
    - Reservation charge
    - GST (applicable on AC classes)
    - Tatkal premium calculation
    - Multi-leg journey cost (with interchange)

  Key outputs:
    - base_fare         (₹)
    - total_fare        (₹, all inclusive)
    - tatkal_fare       (₹)
    - cost_per_km       (₹)
    - value_rating      (Best Value / Good / Expensive)
    - fare_breakdown    (dict of all components)
=============================================================
"""

import math


# ══════════════════════════════════════════════════════════════════════════════
#  IR FARE SLABS (Second Class base, ₹ per km — official 2024 structure)
#  Source: Indian Railways fare table
# ══════════════════════════════════════════════════════════════════════════════

# (max_km, rate_per_km_above_prev_slab)
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
    (9999,  0.28),   # 2000+ km
]

# ── Class multipliers over Second Class base fare ─────────────────────────────
CLASS_MULTIPLIER = {
    "GEN": 1.00,
    "2S" : 1.00,
    "SL" : 1.50,
    "CC" : 2.20,
    "3A" : 2.50,
    "2A" : 3.60,
    "1A" : 5.80,
}

# ── Superfast surcharge (₹, flat per class) ───────────────────────────────────
SUPERFAST_SURCHARGE = {
    "GEN": 15, "2S": 15, "SL": 30, "CC": 45,
    "3A": 45,  "2A": 60, "1A": 75,
}

# ── Reservation charge (₹, flat) ─────────────────────────────────────────────
RESERVATION_CHARGE = {
    "GEN": 0, "2S": 15, "SL": 25, "CC": 40,
    "3A": 40, "2A": 50, "1A": 60,
}

# ── Tatkal multiplier over normal fare (approximate) ─────────────────────────
TATKAL_MULTIPLIER = {
    "SL" : 1.30,
    "CC" : 1.30,
    "3A" : 1.30,
    "2A" : 1.30,
    "1A" : 1.30,
    "2S" : 1.10,
    "GEN": 1.00,
}

# ── GST rate (on AC class base fare only, IR 2024) ───────────────────────────
GST_RATE_AC   = 0.05   # 5% GST on AC classes
AC_CLASSES     = {"1A", "2A", "3A", "CC"}

# ── Train type → superfast? ───────────────────────────────────────────────────
IS_SUPERFAST = {
    "Rajdhani"    : True,
    "Humsafar"    : True,
    "Tejas"       : True,
    "Superfast"   : True,
    "Mail/Express": False,
    "Express"     : False,
    "Passenger"   : False,
}


# ══════════════════════════════════════════════════════════════════════════════
#  FARE CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _base_fare_second_class(distance_km: float) -> float:
    """Calculate Second Class base fare for a given distance using IR slabs."""
    km       = max(1.0, float(distance_km))
    fare     = 0.0
    prev     = 0

    for (max_km, rate) in FARE_SLABS:
        if km <= prev:
            break
        segment = min(km, max_km) - prev
        fare   += segment * rate
        prev    = max_km
        if km <= max_km:
            break

    return max(fare, 25.0)   # minimum fare ₹25


def calculate_fare(
    distance_km  : float,
    travel_class : str,
    train_type   : str  = "Express",
    passengers   : int  = 1,
) -> dict:
    """
    Calculate full fare for one journey segment.

    Args:
        distance_km  : journey distance in km
        travel_class : 1A / 2A / 3A / SL / CC / 2S / GEN
        train_type   : Rajdhani / Superfast / Mail/Express / Passenger
        passengers   : number of passengers

    Returns detailed fare breakdown dict.
    """
    cls      = travel_class.upper()
    ttype    = train_type

    # Base fare (Second Class equivalent)
    base_2nd = _base_fare_second_class(distance_km)

    # Apply class multiplier
    multiplier = CLASS_MULTIPLIER.get(cls, 1.0)
    class_fare = round(base_2nd * multiplier)

    # Superfast surcharge
    superfast     = IS_SUPERFAST.get(ttype, False)
    sf_surcharge  = SUPERFAST_SURCHARGE.get(cls, 0) if superfast else 0

    # Reservation charge
    rsv_charge = RESERVATION_CHARGE.get(cls, 0)

    # Subtotal before GST
    subtotal = class_fare + sf_surcharge + rsv_charge

    # GST (AC classes only)
    gst = round(class_fare * GST_RATE_AC) if cls in AC_CLASSES else 0

    # Total per passenger
    total_per_pax = subtotal + gst

    # Total for all passengers
    total = total_per_pax * passengers

    # Tatkal fare
    tatkal_multiplier = TATKAL_MULTIPLIER.get(cls, 1.3)
    tatkal_extra      = round(class_fare * (tatkal_multiplier - 1))
    tatkal_total      = (total_per_pax + tatkal_extra) * passengers

    # Cost per km
    cost_per_km = round(total_per_pax / max(distance_km, 1), 2)

    # Value rating
    if cost_per_km < 0.8:
        value = "⭐⭐⭐ Best Value"
    elif cost_per_km < 2.0:
        value = "⭐⭐ Good Value"
    elif cost_per_km < 3.5:
        value = "⭐ Fair"
    else:
        value = "💸 Premium"

    return {
        "distance_km"      : distance_km,
        "travel_class"     : cls,
        "train_type"       : ttype,
        "passengers"       : passengers,
        "fare_breakdown"   : {
            "base_fare_2nd_class" : round(base_2nd),
            "class_multiplier"    : multiplier,
            "class_fare"          : class_fare,
            "superfast_surcharge" : sf_surcharge,
            "reservation_charge"  : rsv_charge,
            "gst_5pct"            : gst,
        },
        "fare_per_passenger": total_per_pax,
        "total_fare"        : total,
        "tatkal_extra_per_pax": tatkal_extra,
        "tatkal_total_fare" : tatkal_total,
        "cost_per_km"       : cost_per_km,
        "value_rating"      : value,
        "currency"          : "INR (₹)",
    }


def compare_classes(
    distance_km : float,
    train_type  : str  = "Express",
    passengers  : int  = 1,
) -> list:
    """
    Return fare for all available classes for a given journey,
    sorted by total fare ascending.
    """
    classes     = ["GEN", "2S", "SL", "CC", "3A", "2A", "1A"]
    results     = []

    for cls in classes:
        f = calculate_fare(distance_km, cls, train_type, passengers)
        results.append({
            "class"              : cls,
            "total_fare"         : f["total_fare"],
            "fare_per_passenger" : f["fare_per_passenger"],
            "tatkal_total"       : f["tatkal_total_fare"],
            "cost_per_km"        : f["cost_per_km"],
            "value_rating"       : f["value_rating"],
        })

    results.sort(key=lambda x: x["total_fare"])
    return results


def calculate_route_cost(
    route       : dict,
    travel_class: str = "SL",
    passengers  : int = 1,
) -> dict:
    """
    Calculate total cost for a full route from route_engine output.
    Handles multi-leg journeys (with interchanges).

    Usage:
        result  = graph.smart_route_search("KOAA", "GHY")
        for r in result["routes"]:
            cost = calculate_route_cost(r, travel_class="3A", passengers=2)
    """
    legs        = route.get("legs", [])
    train_type  = route.get("train_type", route.get("train_category", "Express"))
    total_cost  = 0
    leg_costs   = []

    for leg in legs:
        dist  = leg.get("distance_km", 0) or route.get("total_distance_km", 500)
        ttype = leg.get("train_category", train_type)

        f = calculate_fare(dist, travel_class, ttype, passengers)
        total_cost += f["total_fare"]
        leg_costs.append({
            "leg"          : leg.get("leg", 1),
            "from"         : leg.get("from_name", leg.get("from", "")),
            "to"           : leg.get("to_name",   leg.get("to", "")),
            "train"        : leg.get("train_number"),
            "distance_km"  : dist,
            "fare"         : f["total_fare"],
            "tatkal_fare"  : f["tatkal_total_fare"],
        })

    # If no legs had distance, estimate from route total
    if total_cost == 0:
        total_dist = route.get("total_distance_km", 500)
        f = calculate_fare(total_dist, travel_class, train_type, passengers)
        total_cost = f["total_fare"]
        tatkal_cost = f["tatkal_total_fare"]
    else:
        tatkal_cost = sum(l["tatkal_fare"] for l in leg_costs)

    return {
        "travel_class"   : travel_class,
        "passengers"     : passengers,
        "leg_costs"      : leg_costs,
        "total_fare"     : round(total_cost),
        "tatkal_fare"    : round(tatkal_cost),
        "currency"       : "INR (₹)",
        "note"           : (
            f"₹{round(total_cost/passengers)} per person"
            if passengers > 1 else ""
        ),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  CostService — Test")
    print("=" * 55)

    # Test single fare
    fare = calculate_fare(
        distance_km=1144,
        travel_class="SL",
        train_type="Mail/Express",
        passengers=2,
    )
    print(f"\n  KOAA → GHY  |  SL  |  2 passengers  |  1144 km")
    print(f"    Fare breakdown : {fare['fare_breakdown']}")
    print(f"    Per passenger  : ₹{fare['fare_per_passenger']}")
    print(f"    Total fare     : ₹{fare['total_fare']}")
    print(f"    Tatkal total   : ₹{fare['tatkal_total_fare']}")
    print(f"    Cost/km        : ₹{fare['cost_per_km']}")
    print(f"    Value          : {fare['value_rating']}")

    # Compare classes
    print(f"\n  Class comparison for 1144 km (Express, 1 passenger):")
    print(f"  {'Class':<6} {'Fare':>8} {'Tatkal':>10} {'₹/km':>8} {'Value'}")
    print(f"  {'-'*55}")
    for c in compare_classes(1144, "Express", 1):
        print(f"  {c['class']:<6} ₹{c['fare_per_passenger']:>6} ₹{c['tatkal_total']:>8}  "
              f"₹{c['cost_per_km']:>5}  {c['value_rating']}")

    # Superfast test
    print(f"\n  Rajdhani fare  |  2A  |  1 passenger  |  1400 km:")
    f2 = calculate_fare(1400, "2A", "Rajdhani", 1)
    print(f"    Total fare : ₹{f2['total_fare']}")
    print(f"    Tatkal     : ₹{f2['tatkal_total_fare']}")
    print(f"    Value      : {f2['value_rating']}")