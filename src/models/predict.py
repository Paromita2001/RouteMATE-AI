# """
# =============================================================
#   RouteMATE_AI — Predict
#   File: src/models/predict.py
# =============================================================
#   Thin, clean prediction API that wires together:
#     - route_engine   → find candidate routes
#     - DelayPredictor → score each route with ML
#     - weather_service→ live weather risk
#     - calendar_service→ date demand & holiday context
#     - availability_service → seat availability
#     - cost_service   → fare breakdown

#   Single entry point used by app.py:
#     from src.models.predict import RoutePredictor
#     predictor = RoutePredictor()
#     result    = predictor.predict(
#                     origin="KOAA", destination="GHY",
#                     travel_date=date(2025,11,2),
#                     travel_class="SL", passengers=2)
# =============================================================
# """

# import os
# import sys
# import datetime
# import warnings
# warnings.filterwarnings("ignore")

# ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if ROOT not in sys.path:
#     sys.path.insert(0, ROOT)

# from src.graph.route_engine            import RouteGraph
# from src.models.train_model            import DelayPredictor
# from src.services.weather_service      import get_route_weather
# from src.services.calendar_service     import get_calendar_info, is_tatkal_recommended
# from src.services.availability_service import get_all_class_availability
# from src.services.cost_service         import calculate_fare, compare_classes
# from src.utils.helpers                 import (
#     format_duration, format_inr, compute_route_score, route_score_label,
#     standardize_station_code, validate_station_code,
#     validate_travel_date, validate_passenger_count,
# )
# from src.utils.constants import MODEL_PKL, ROUTES_CSV, DELAY_CSV


# class RoutePredictor:
#     """
#     Full prediction pipeline.
#     Initialise once at app startup, then call .predict() per query.
#     """

#     def __init__(
#         self,
#         model_path : str = MODEL_PKL,
#         routes_csv : str = ROUTES_CSV,
#         delay_csv  : str = DELAY_CSV,
#     ):
#         print("🚀 Initialising RouteMATE AI predictor...")
#         self.graph     = RouteGraph()
#         self.graph.build(routes_csv, delay_csv)
#         self.predictor = DelayPredictor(model_path)
#         self._station_codes = set(self.graph.G.nodes())
#         print("✅ Predictor ready.\n")

#     # ── Main predict ─────────────────────────────────────────────────────────

#     def predict(
#         self,
#         origin        : str,
#         destination   : str,
#         travel_date   : datetime.date = None,
#         travel_class  : str           = "SL",
#         passengers    : int           = 1,
#         top_n         : int           = 3,
#         max_changes   : int           = 1,
#     ) -> dict:
#         """
#         Full end-to-end prediction for a journey.

#         Returns dict with:
#             query, validation, context, routes,
#             best_route, fare_options, availability, summary
#         """
#         if travel_date is None:
#             travel_date = datetime.date.today()

#         origin      = standardize_station_code(origin)
#         destination = standardize_station_code(destination)

#         # ── Validation ────────────────────────────────────────────────────────
#         errors = []
#         ok, msg = validate_station_code(origin, self._station_codes)
#         if not ok: errors.append(msg)
#         ok, msg = validate_station_code(destination, self._station_codes)
#         if not ok: errors.append(msg)
#         ok, msg = validate_travel_date(travel_date)
#         if not ok: errors.append(msg)
#         ok, msg = validate_passenger_count(passengers)
#         if not ok: errors.append(msg)
#         if origin == destination:
#             errors.append("Origin and destination cannot be the same station.")

#         if errors:
#             return {
#                 "query"      : self._build_query(origin, destination, travel_date, travel_class, passengers),
#                 "validation" : {"valid": False, "errors": errors},
#                 "routes"     : [], "best_route": None,
#                 "context"    : None, "fare_options": None,
#                 "availability": None,
#                 "summary"    : "❌ " + " | ".join(errors),
#             }

#         # ── Calendar + Weather context ─────────────────────────────────────────
#         calendar = get_calendar_info(travel_date)
#         occ      = calendar["occupancy_factor"]
#         cal_boost= calendar["delay_risk_boost"]

#         corridor = self._get_corridor_stations(origin, destination)
#         weather  = get_route_weather(corridor, travel_date)
#         w_factor = weather["max_delay_risk_factor"]

#         # ── Route search + ML scoring ─────────────────────────────────────────
#         raw    = self.graph.smart_route_search(origin, destination,
#                                                top_n=top_n, max_changes=max_changes)
#         if not raw["routes"]:
#             return {
#                 "query"      : self._build_query(origin, destination, travel_date, travel_class, passengers),
#                 "validation" : {"valid": True, "errors": []},
#                 "routes"     : [], "best_route": None,
#                 "context"    : self._build_context(calendar, weather),
#                 "fare_options": None, "availability": None,
#                 "summary"    : f"No routes found between {origin} and {destination}.",
#             }

#         scored = self.predictor.score_all_routes(raw)

#         # ── Enrich each route with context-adjusted scores and fares ──────────
#         enriched = []
#         for route in scored["routes"]:
#             base_delay = route.get("predicted_avg_delay_min", 0) or 0
#             adj_delay  = (base_delay + cal_boost) * w_factor
#             adj_score  = compute_route_score(adj_delay, route.get("changes", 0),
#                                               w_factor, occ)
#             fare = calculate_fare(
#                 #distance_km  = route.get("total_distance_km") or 500,
#                 distance_km = route.get("total_distance_km")
#                 if not distance_km:
#                     continue
#                 travel_class = travel_class,
#                 train_type   = route.get("train_category", "Express"),
#                 passengers   = passengers,
#             )
#             route["adjusted_delay_min"] = round(adj_delay, 1)
#             route["adjusted_score"]     = adj_score
#             route["adjusted_label"]     = route_score_label(adj_score)
#             route["fare_estimate"]      = fare
#             enriched.append(route)

#         enriched.sort(key=lambda r: r["adjusted_score"], reverse=True)
#         best = enriched[0] if enriched else None

#         # ── Fare options + availability for best route ─────────────────────────
#         fare_options = availability = None
#         if best:
#            # dist  = best.get("total_distance_km") or 500
#             dist = best.get("total_distance_km")
#             if not dist:
#                 dist = 0
#             ttype = best.get("train_category", "Express")
#             fare_options = compare_classes(dist, ttype, passengers)
#             availability = get_all_class_availability(
#                 train_number     = best.get("train_number") or 0,
#                 train_type       = ttype,
#                 travel_date      = travel_date,
#                 distance_km      = dist,
#                 occupancy_factor = occ,
#             )

#         tatkal_rec, tatkal_reason = is_tatkal_recommended(travel_date)

#         return {
#             "query"              : self._build_query(origin, destination, travel_date,
#                                                       travel_class, passengers),
#             "validation"         : {"valid": True, "errors": []},
#             "context"            : self._build_context(calendar, weather),
#             "routes"             : enriched,
#             "best_route"         : best,
#             "fare_options"       : fare_options,
#             "availability"       : availability,
#             "tatkal_recommended" : tatkal_rec,
#             "tatkal_reason"      : tatkal_reason,
#             "summary"            : self._build_summary(best, travel_class, passengers, tatkal_rec),
#         }

#     # ── Helpers ───────────────────────────────────────────────────────────────

#     # def _get_corridor_stations(self, origin, destination):
#     #     from src.utils.constants import STATION_COORDINATES
#     #     known = list(STATION_COORDINATES.keys())
#     #     return [origin, destination] + [s for s in known
#     #                                      if s not in (origin, destination)][:8]
  
#     def _get_corridor_stations(self, origin, destination):
#         try:
#             path = self.graph.shortest_path(origin, destination)
#             return path if path else [origin, destination]
#         except Exception:
#             return [origin, destination]

#     def _build_query(self, origin, destination, travel_date, travel_class, passengers):
#         return {
#             "origin"          : origin,
#             "origin_name"     : self.graph.get_station_name(origin),
#             "destination"     : destination,
#             "destination_name": self.graph.get_station_name(destination),
#             "travel_date"     : str(travel_date),
#             "travel_class"    : travel_class,
#             "passengers"      : passengers,
#         }

#     def _build_context(self, calendar, weather):
#         return {
#             "calendar"           : calendar,
#             "weather"            : weather,
#             "occupancy_factor"   : calendar["occupancy_factor"],
#             "weather_risk_factor": weather["max_delay_risk_factor"],
#             "travel_advisory"    : calendar["travel_advisory"],
#             "weather_advisories" : weather.get("route_advisories", []),
#             "demand_label"       : calendar["risk_label"],
#             "worst_weather"      : weather["worst_condition"],
#         }

#     def _build_summary(self, best, travel_class, passengers, tatkal_rec):
#         if not best:
#             return "No routes found."
#         hrs   = best.get("total_travel_hrs", 0) or 0
#         delay = best.get("adjusted_delay_min", 0) or 0
#         score = best.get("adjusted_score", 0)
#         label = best.get("adjusted_label", "")
#         fare  = best.get("fare_estimate", {}).get("total_fare", 0)
#         train = best.get("train_name", "Unknown")
#         risk  = best.get("overall_risk", "")
#         tatkal = " ⚠️ Tatkal recommended." if tatkal_rec else ""
#         return (
#             f"Best: {train} | {hrs:.1f} hrs | Delay ~{delay:.0f} min | "
#             f"Risk {risk} | Score {score} {label} | "
#             f"Fare {format_inr(fare)} ({travel_class}, {passengers} pax).{tatkal}"
#         )

#     def get_station_list(self):
#         return sorted(self._station_codes)

#     def station_search(self, query: str) -> list:
#         q = query.upper().strip()
#         results = []
#         for code in self._station_codes:
#             name = self.graph.get_station_name(code)
#             if q in code or q in name.upper():
#                 results.append({"code": code, "name": name,
#                                  "label": f"{name} ({code})"})
#         return sorted(results, key=lambda x: x["code"])[:20]


# # ── Quick test ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     rp = RoutePredictor()

#     result = rp.predict(
#         origin      = "KOAA",
#         destination = "GHY",
#         travel_date = datetime.date.today() + datetime.timedelta(days=7),
#         travel_class= "SL",
#         passengers  = 2,
#     )

#     print(f"Valid     : {result['validation']['valid']}")
#     print(f"Routes    : {len(result['routes'])}")
#     print(f"Calendar  : {result['context']['travel_advisory']}")
#     print(f"Weather   : {result['context']['worst_weather']}")
#     for i, r in enumerate(result["routes"], 1):
#         print(f"  Route {i}: {r['train_name']} | "
#               f"{r['total_travel_hrs']:.1f}h | "
#               f"delay ~{r['adjusted_delay_min']:.0f}min | "
#               f"score {r['adjusted_score']} | "
#               f"{format_inr(r['fare_estimate']['total_fare'])}")
#     print(f"\nSummary: {result['summary']}")




import os
import sys
import datetime
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph.route_engine import RouteGraph
from src.models.train_model import DelayPredictor
from src.services.weather_service import get_route_weather
from src.services.calendar_service import get_calendar_info, is_tatkal_recommended
from src.services.availability_service import get_all_class_availability
from src.services.cost_service import calculate_fare, compare_classes
from src.utils.helpers import (
    format_inr,
    compute_route_score,
    route_score_label,
    standardize_station_code,
    validate_station_code,
    validate_travel_date,
    validate_passenger_count,
)
from src.utils.constants import MODEL_PKL, ROUTES_CSV, DELAY_CSV

def _get_route_distance(route: dict) -> float:
    # Try direct field
    d = route.get("total_distance_km")
    if d and d > 0:
        return float(d)

    # Try summing legs
    legs = route.get("legs", [])
    total = sum(float(leg.get("distance_km") or 0) for leg in legs)
    if total > 0:
        return total

    # Fallback (estimate)
    mins = route.get("total_travel_min") or 600
    return max(mins * 0.8, 100)

class RoutePredictor:

    def __init__(self, model_path=MODEL_PKL, routes_csv=ROUTES_CSV, delay_csv=DELAY_CSV):
        logging.info("🚀 Initialising RouteMATE AI predictor...")

        self.graph = RouteGraph()
        self.graph.build(routes_csv, delay_csv)

        self.predictor = DelayPredictor(model_path)
        self._station_codes = set(self.graph.G.nodes())

        logging.info("✅ Predictor ready.\n")

    # ========================= MAIN FUNCTION =========================

    def predict(
        self,
        origin,
        destination,
        travel_date=None,
        travel_class="SL",
        passengers=1,
        top_n=3,
        max_changes=1,
    ):

        if travel_date is None:
            travel_date = datetime.date.today()

        origin = standardize_station_code(origin)
        destination = standardize_station_code(destination)

        # ---------------- VALIDATION ----------------
        errors = []

        for code in [origin, destination]:
            ok, msg = validate_station_code(code, self._station_codes)
            if not ok:
                errors.append(msg)

        ok, msg = validate_travel_date(travel_date)
        if not ok:
            errors.append(msg)

        ok, msg = validate_passenger_count(passengers)
        if not ok:
            errors.append(msg)

        if origin == destination:
            errors.append("Origin and destination cannot be same.")

        if errors:
            # return {
            #     "validation": {"valid": False, "errors": errors},
            #     "routes": [],
            #     "summary": " | ".join(errors),
            # }
            return {
                "query": {
                    "origin": origin,
                    "origin_name": self.graph.get_station_name(origin),
                    "destination": destination,
                    "destination_name": self.graph.get_station_name(destination),
                    "travel_date": str(travel_date),
                    "travel_class": travel_class,
                    "passengers": passengers,
                },
                "validation": {"valid": True, "errors": errors},
                "routes": [],
                "best_route": None,
                "context": None,
                "fare_options": None,
                "availability": None,
                "summary": "No routes found",
            }

        # ---------------- CONTEXT ----------------
        calendar = get_calendar_info(travel_date)
        occ = calendar["occupancy_factor"]
        cal_boost = calendar["delay_risk_boost"]

        corridor = self._get_corridor_stations(origin, destination)
        weather = get_route_weather(corridor, travel_date)
        w_factor = weather["max_delay_risk_factor"]

        # ---------------- ROUTE SEARCH ----------------
        raw = self.graph.smart_route_search(origin, destination, top_n=top_n, max_changes=max_changes)

        if not raw["routes"]:
            # return {
            #     "validation": {"valid": True, "errors": []},
            #     "routes": [],
            #     "summary": "No routes found",
            # }
            return {
                "query": {
                    "origin": origin,
                    "origin_name": self.graph.get_station_name(origin),
                    "destination": destination,
                    "destination_name": self.graph.get_station_name(destination),
                    "travel_date": str(travel_date),
                    "travel_class": travel_class,
                    "passengers": passengers,
                },
                "validation": {"valid": True, "errors": []},
                "routes": [],
                "best_route": None,
                "context": None,
                "fare_options": None,
                "availability": None,
                "summary": f"No routes found between {origin} and {destination}.",
            }

        # ---------------- ML SCORING ----------------
        try:
            scored = self.predictor.score_all_routes(raw)
        except Exception as e:
            # return {
            #     "validation": {"valid": False, "errors": [str(e)]},
            #     "routes": [],
            #     "summary": f"Model error: {str(e)}",
            # }
            return {
                "query": {
                    "origin": origin,
                    "origin_name": self.graph.get_station_name(origin),
                    "destination": destination,
                    "destination_name": self.graph.get_station_name(destination),
                    "travel_date": str(travel_date),
                    "travel_class": travel_class,
                    "passengers": passengers,
                },
                "validation": {"valid": True, "errors": []},
                "routes": [],
                "best_route": None,
                "context": None,
                "fare_options": None,
                "availability": None,
                "summary": "No routes found",
            }

        # ---------------- ENRICH ROUTES ----------------
        enriched = []

        for route in scored["routes"]:

            base_delay = route.get("predicted_avg_delay_min", 0) or 0

            adj_delay = (base_delay + cal_boost) * w_factor

            adj_score = compute_route_score(
                adj_delay,
                route.get("changes", 0),
                w_factor,
                occ,
            )

            # distance_km = route.get("total_distance_km")

            # if not distance_km or distance_km <= 0:
            #     continue  # skip invalid route

            # fare = calculate_fare(
            #     distance_km=distance_km,
            #     travel_class=travel_class,
            #     train_type=route.get("train_category", "Express"),
            #     passengers=passengers,
            # )



            dist_km = _get_route_distance(route)


            if dist_km <= 0:
                dist_km = 100  # fallback minimum

            route["total_distance_km"] = dist_km

            fare = calculate_fare(
                distance_km=dist_km,
                travel_class=travel_class,
                train_type=route.get("train_category", "Express"),
                passengers=passengers,
            )

            # ---------- FINAL OUTPUT ----------
            route["adjusted_delay_min"] = round(adj_delay, 1)
            route["adjusted_score"] = adj_score
            route["adjusted_label"] = route_score_label(adj_score)
            route["fare_estimate"] = fare

            # ---------- SCORE BREAKDOWN ----------
            route["score_breakdown"] = {
                "base_delay": base_delay,
                "calendar_boost": cal_boost,
                "weather_factor": w_factor,
                "final_delay": adj_delay,
                "changes": route.get("changes", 0),
            }

            # ---------- CONFIDENCE ----------
            route["confidence"] = round(1 - (adj_delay / 300), 2)

            enriched.append(route)

        enriched.sort(key=lambda r: r["adjusted_score"], reverse=True)

        best = enriched[0] if enriched else None

        # ---------- FIX RISK LABEL ----------
        if best:
            best["overall_risk"] = route_score_label(best["adjusted_score"])

        # ---------------- FARE + AVAILABILITY ----------------
        fare_options = availability = None

        if best:
            dist = best.get("total_distance_km") or 0
            ttype = best.get("train_category", "Express")

            fare_options = compare_classes(dist, ttype, passengers)

            availability = get_all_class_availability(
                train_number=best.get("train_number"),
                train_type=ttype,
                travel_date=travel_date,
                distance_km=dist,
                occupancy_factor=occ,
            )

        tatkal_rec, tatkal_reason = is_tatkal_recommended(travel_date)

        # return {
        #     "validation": {"valid": True, "errors": []},
        #     "routes": enriched,
        #     "best_route": best,
        #     "fare_options": fare_options,
        #     "availability": availability,
        #     "tatkal_recommended": tatkal_rec,
        #     "tatkal_reason": tatkal_reason,
        #     "summary": self._build_summary(best, travel_class, passengers, tatkal_rec),
        # }

        return {
            "query": {
                "origin": origin,
                "origin_name": self.graph.get_station_name(origin),
                "destination": destination,
                "destination_name": self.graph.get_station_name(destination),
                "travel_date": str(travel_date),
                "travel_class": travel_class,
                "passengers": passengers,
            },
            "validation": {"valid": True, "errors": []},
            "routes": enriched,
            "best_route": best,
            "fare_options": fare_options,
            "availability": availability,
            "tatkal_recommended": tatkal_rec,
            "tatkal_reason": tatkal_reason,
            "summary": self._build_summary(best, travel_class, passengers, tatkal_rec),
        }

    # ========================= HELPERS =========================

    def _get_corridor_stations(self, origin, destination):
        try:
            path = self.graph.shortest_path(origin, destination)
            return path if path else [origin, destination]
        except:
            return [origin, destination]

    def _build_summary(self, best, travel_class, passengers, tatkal_rec):
        if not best:
            return "No routes found"

        return (
            f"{best.get('train_name')} | "
            f"{best.get('total_travel_hrs', 0):.1f} hrs | "
            f"Delay ~{best.get('adjusted_delay_min', 0):.0f} min | "
            f"Score {best.get('adjusted_score')} | "
            f"Fare {format_inr(best.get('fare_estimate', {}).get('total_fare', 0))}"
        )