from src.models.train_model import DelayPredictor
from src.services.weather_service import get_route_weather
#from src.services.calendar_service import get_calendar_info
from src.services.cost_service import calculate_route_cost
from src.services.availability_service import get_availability

import datetime
import src.services.calendar_service as cal

class SmartRouteIntelligence:

    def __init__(self):
        self.predictor = DelayPredictor()

    def analyze_route(self, route: dict, travel_date=None):

        if travel_date is None:
            travel_date = datetime.date.today()

        # -------------------------
        # 1. ML Delay Prediction
        # -------------------------
        route = self.predictor.score_route(route)

        # -------------------------
        # 2. Weather
        # -------------------------
        station_codes = [leg["to"] for leg in route["legs"]]
        weather = get_route_weather(station_codes, travel_date)

        # -------------------------
        # 3. Calendar
        # -------------------------
        calendar = cal.get_calendar_info(travel_date)

        # -------------------------
        # 4. Cost
        # -------------------------
        cost = calculate_route_cost(route, travel_class="SL", passengers=1)

        # -------------------------
        # 5. Availability (first leg)
        # -------------------------
        first_leg = route["legs"][0]

        availability = get_availability(
            train_number=first_leg["train_number"],
            train_type=first_leg["train_category"],
            travel_class="SL",
            travel_date=travel_date,
            distance_km=first_leg.get("distance_km", 500),
            occupancy_factor=calendar["occupancy_factor"],
        )

        # -------------------------
        # FINAL SCORE
        # -------------------------
        score = route["route_score"]

        # weather impact
        score -= (weather["max_delay_risk_factor"] - 1) * 10

        # calendar impact
        score -= (calendar["occupancy_factor"] - 1) * 15

        # availability impact
        if availability["availability_status"] == "Waitlist":
            score -= 20

        return {
            "route": route,
            "weather": weather,
            "calendar": calendar,
            "cost": cost,
            "availability": availability,
            "final_score": round(score, 2)
        }