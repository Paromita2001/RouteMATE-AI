"""
=============================================================
  RouteMATE_AI — Graph Route Engine
  File: src/graph/route_engine.py
=============================================================
  Builds a weighted directed graph from train route data.

  Graph structure:
    Nodes  = stations (station_code)
    Edges  = direct train connection between two consecutive stops
    Weight = travel time in minutes (or distance as fallback)

  Key features:
    ✅ Build graph from train_routes_clean.csv
    ✅ Find direct routes between two stations
    ✅ Find routes with 1 interchange (multi-train journeys)
    ✅ Rank routes by travel time / distance / changes
    ✅ Return top N smart routes with full details
    ✅ Ready for ML delay injection in next step
=============================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import networkx as nx
from itertools import islice

warnings.filterwarnings("ignore")

# ── Path (relative to project root) ──────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROUTES_CSV    = os.path.join(BASE_DIR, "data", "processed", "train_routes_clean.csv")
DELAY_CSV     = os.path.join(BASE_DIR, "data", "processed", "merged_delay.csv")

# ── Dev/testing fallback paths (comment out in production) ───────────────────
if not os.path.exists(ROUTES_CSV):
    ROUTES_CSV = "/mnt/user-data/outputs/preprocessed/train_routes_clean.csv"
    DELAY_CSV  = "/mnt/user-data/outputs/preprocessed/merged_delay.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

class RouteGraph:
    """
    Directed weighted graph of the Indian Railway network.

    Each edge carries:
        train_number, train_name, train_category,
        depart_time, arrive_time, travel_min, distance_km,
        running_days
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()          # MultiDiGraph: multiple trains on same edge
        self.station_meta = {}              # code → full name
        self.delay_meta   = {}              # code → avg delay info
        self._built = False

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self, routes_csv: str = ROUTES_CSV, delay_csv: str = DELAY_CSV):
        """
        Load processed CSVs and construct the graph.
        Call this once before any queries.
        """
        print("📦 Loading route data...")
        df = pd.read_csv(routes_csv)
        print(f"   {len(df):,} stop records | {df['train_number'].nunique():,} trains | "
              f"{df['station_code'].nunique():,} stations")

        # --- Load delay data for node enrichment ---
        try:
            delay_df = pd.read_csv(delay_csv)
            for _, row in delay_df.iterrows():
                self.delay_meta[row["station_code"]] = {
                    "avg_delay_min"   : row["avg_delay_min"],
                    "delay_category"  : row["delay_category"],
                    "pct_right_time"  : row["pct_right_time"],
                }
        except Exception as e:
            print(f"   ⚠️  Could not load delay data: {e}")

        # --- Add nodes (stations) ---
        print("🔵 Adding station nodes...")
        for code, grp in df.groupby("station_code"):
            name = grp["station_name"].iloc[0]
            self.station_meta[code] = name
            self.G.add_node(
                code,
                station_name  = name,
                avg_delay_min = self.delay_meta.get(code, {}).get("avg_delay_min", np.nan),
                delay_category= self.delay_meta.get(code, {}).get("delay_category", "Unknown"),
                pct_right_time= self.delay_meta.get(code, {}).get("pct_right_time", np.nan),
            )

        # --- Build edges (consecutive stops per train) ---
        print("🔗 Building edges between consecutive stops...")
        df_sorted = df.sort_values(["train_number", "stop_no"])

        edge_count = 0
        for train_num, train_df in df_sorted.groupby("train_number"):
            stops = train_df.reset_index(drop=True)
            meta = {
                "train_number"   : train_num,
                "train_name"     : stops["train_name"].iloc[0],
                "train_category" : stops["train_category"].iloc[0],
                "running_days"   : stops["running_days"].iloc[0],
            }

            for i in range(len(stops) - 1):
                src = stops.loc[i,   "station_code"]
                dst = stops.loc[i+1, "station_code"]

                dep_min  = stops.loc[i,   "departs_min"]   # depart from src
                arr_min  = stops.loc[i+1, "arrives_min"]   # arrive at dst
                dist_km  = stops.loc[i+1, "distance_km"] - stops.loc[i, "distance_km"]

                # Travel time (handle overnight trains where arr < dep)
                if pd.notna(dep_min) and pd.notna(arr_min):
                    travel_min = arr_min - dep_min
                    if travel_min < 0:
                        travel_min += 1440          # crossed midnight
                else:
                    travel_min = np.nan

                # Edge weight: prefer travel_min, fallback to distance
                weight = travel_min if pd.notna(travel_min) else max(dist_km, 1)

                self.G.add_edge(
                    src, dst,
                    weight       = weight,
                    travel_min   = travel_min,
                    distance_km  = max(dist_km, 0),
                    depart_time  = stops.loc[i,   "departs"],
                    arrive_time  = stops.loc[i+1, "arrives"],
                    depart_min   = dep_min,
                    arrive_min   = arr_min,
                    day_src      = int(stops.loc[i,   "day"]),
                    day_dst      = int(stops.loc[i+1, "day"]),
                    **meta,
                )
                edge_count += 1






        # 🔥 ADD THIS BLOCK HERE
        print("📇 Building train stop index...")

        df_sorted = df.sort_values(["train_number", "stop_no"])
        self._train_stops = {}

        for train_num, train_df in df_sorted.groupby("train_number"):
            stops = train_df.reset_index(drop=True)

            self._train_stops[train_num] = [
                {
                    "station": row.station_code,
                    "departs_min": row.departs_min,
                    "arrives_min": row.arrives_min,
                    "distance_km": row.distance_km,
                    "day": row.day,
                    "train_name": row.train_name,
                    "train_category": row.train_category,
                    "running_days": row.running_days,
                    "departs": row.departs,
                    "arrives": row.arrives,
                }
                for row in stops.itertuples()
            ]







        self._built = True
        print(f"✅ Graph built: {self.G.number_of_nodes():,} nodes | "
              f"{self.G.number_of_edges():,} edges | {edge_count:,} train connections")
        return self


    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check_built(self):
        if not self._built:
            raise RuntimeError("Graph not built. Call .build() first.")

    def _edge_summary(self, u, v, edge_data: dict) -> dict:
        """Format a single edge into a clean stop dict."""
        return {
            "from_code"     : u,
            "from_name"     : self.station_meta.get(u, u),
            "to_code"       : v,
            "to_name"       : self.station_meta.get(v, v),
            "train_number"  : edge_data.get("train_number"),
            "train_name"    : edge_data.get("train_name"),
            "train_category": edge_data.get("train_category"),
            "running_days"  : edge_data.get("running_days"),
            "depart_time"   : edge_data.get("depart_time"),
            "arrive_time"   : edge_data.get("arrive_time"),
            "travel_min"    : edge_data.get("travel_min"),
            "distance_km"   : edge_data.get("distance_km"),
        }

    def _format_time(self, minutes) -> str:
        """Convert float minutes → 'HH:MM' string."""
        if pd.isna(minutes):
            return "N/A"
        h = int(minutes) // 60 % 24
        m = int(minutes) % 60
        return f"{h:02d}:{m:02d}"

    def _risk_label(self, avg_delay: float) -> str:
        if pd.isna(avg_delay):
            return "Unknown"
        if avg_delay <= 15:
            return "🟢 Low"
        elif avg_delay <= 45:
            return "🟡 Medium"
        else:
            return "🔴 High"


    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def station_exists(self, code: str) -> bool:
        return code.upper() in self.G.nodes

    def get_station_name(self, code: str) -> str:
        return self.station_meta.get(code.upper(), code)

    def get_station_delay_info(self, code: str) -> dict:
        node = self.G.nodes.get(code.upper(), {})
        return {
            "station_code"  : code.upper(),
            "station_name"  : node.get("station_name", code),
            "avg_delay_min" : node.get("avg_delay_min", np.nan),
            "delay_category": node.get("delay_category", "Unknown"),
            "pct_right_time": node.get("pct_right_time", np.nan),
            "risk_label"    : self._risk_label(node.get("avg_delay_min", np.nan)),
        }

    def find_direct_routes(self, origin: str, destination: str) -> list[dict]:
        """
        Find all trains that run DIRECTLY from origin → destination
        (i.e., origin and destination are both stops on the same train,
        in the correct order).

        Returns list of route dicts sorted by total travel time.
        """
        self._check_built()
        origin, destination = origin.upper(), destination.upper()

        if not self.station_exists(origin):
            return [{"error": f"Station '{origin}' not found in graph."}]
        if not self.station_exists(destination):
            return [{"error": f"Station '{destination}' not found in graph."}]

        results = []







        

        # 🔥 NEW LOGIC USING TRAIN STOPS
        for train_num, stops in self._train_stops.items():

            # map station → index
            station_to_idx = {}
            for i, s in enumerate(stops):
                if s["station"] not in station_to_idx:
                    station_to_idx[s["station"]] = i

            # skip if train doesn't cover both stations
            if origin not in station_to_idx or destination not in station_to_idx:
                continue

            o_idx = station_to_idx[origin]
            d_idx = station_to_idx[destination]

            # ensure correct direction
            if d_idx <= o_idx:
                continue

            o_stop = stops[o_idx]
            d_stop = stops[d_idx]

            # ✅ correct distance
            total_dist = max(d_stop["distance_km"] - o_stop["distance_km"], 0)

            # ✅ correct time
            dep = o_stop["departs_min"]
            arr = d_stop["arrives_min"]

            if pd.notna(dep) and pd.notna(arr):
                total_min = arr - dep
                if total_min < 0:
                    total_min += 1440
            else:
                total_min = np.nan

            delay_info = self.get_station_delay_info(origin)

            results.append({
                "route_type": "Direct",
                "changes": 0,
                "train_number": train_num,
                "train_name": o_stop["train_name"],
                "train_category": o_stop["train_category"],
                "running_days": o_stop["running_days"],
                "origin": origin,
                "origin_name": self.get_station_name(origin),
                "destination": destination,
                "destination_name": self.get_station_name(destination),
                "total_travel_min": total_min,
                "total_travel_hrs": round(total_min / 60, 2) if pd.notna(total_min) else None,
                "total_distance_km": total_dist,
                "origin_delay_min": delay_info["avg_delay_min"],
                "origin_risk": delay_info["risk_label"],
                "legs": [{
                    "leg": 1,
                    "from": origin,
                    "to": destination,
                    "train_number": train_num,
                    "train_name": o_stop["train_name"],
                    "travel_min": total_min,
                    "distance_km": total_dist,
                }],
            })

        # # Collect all trains passing through origin
        # trains_at_origin = set()
        # for _, dst, edata in self.G.out_edges(origin, data=True):
        #     trains_at_origin.add(edata["train_number"])

        # # For each such train, check if destination is a later stop
        # for train_num in trains_at_origin:
        #     # Get all edges for this train
        #     # train_edges = [
        #     #     (u, v, d) for u, v, d in self.G.edges(data=True)
        #     #     if d.get("train_number") == train_num
        #     # ]
            # if not train_edges:
            #     continue

            # # Build ordered stop sequence
            # stop_order = {}
            # for u, v, d in train_edges:
            #     if u not in stop_order:
            #         stop_order[u] = d.get("depart_min", np.nan)

            # # Check if destination appears after origin in this train's path
            # origin_dep  = stop_order.get(origin, np.nan)
            # dest_arr    = None

            # # Find the edge that arrives at destination on this train
            # for u, v, d in train_edges:
            #     if v == destination:
            #         dest_arr = d.get("arrive_min", np.nan)
            #         break

            # if dest_arr is None:
            #     continue  # destination not on this train

            # # Compute total travel time
            # if pd.notna(origin_dep) and pd.notna(dest_arr):
            #     total_min = dest_arr - origin_dep
            #     if total_min < 0:
            #         total_min += 1440
            # else:
            #     total_min = np.nan

            # # Get total distance
            # # dist_edges = [d for u, v, d in train_edges]
            # # total_dist = sum(d.get("distance_km", 0) for d in dist_edges)
            
            # total_dist = 0
            # start_collecting = False

            # for u, v, d in train_edges:
            #     if u == origin:
            #         start_collecting = True

            #     if start_collecting:
            #         total_dist += d.get("distance_km", 0)

            #     if v == destination:
            #         break

            # meta = train_edges[0][2]  # first edge for metadata

            # # Delay risk at origin
            # delay_info = self.get_station_delay_info(origin)

            # results.append({
            #     "route_type"     : "Direct",
            #     "changes"        : 0,
            #     "train_number"   : train_num,
            #     "train_name"     : meta.get("train_name"),
            #     "train_category" : meta.get("train_category"),
            #     "running_days"   : meta.get("running_days"),
            #     "origin"         : origin,
            #     "origin_name"    : self.get_station_name(origin),
            #     "destination"    : destination,
            #     "destination_name": self.get_station_name(destination),
            #     "total_travel_min"   : total_min,
            #     "total_travel_hrs"   : round(total_min / 60, 2) if pd.notna(total_min) else None,
            #     "origin_delay_min"   : delay_info["avg_delay_min"],
            #     "origin_risk"        : delay_info["risk_label"],
            #     "legs"           : [{
            #         "leg"          : 1,
            #         "from"         : origin,
            #         "from_name"    : self.get_station_name(origin),
            #         "to"           : destination,
            #         "to_name"      : self.get_station_name(destination),
            #         "train_number" : train_num,
            #         "train_name"   : meta.get("train_name"),
            #         "train_category": meta.get("train_category"),
            #         "running_days" : meta.get("running_days"),
            #         "depart_time"  : None,
            #         "arrive_time"  : None,
            #         "travel_min"   : total_min,
            #         "distance_km"  : total_dist,
            #     }],
            # })

        # Sort by travel time
        results.sort(key=lambda x: x["total_travel_min"] if pd.notna(x.get("total_travel_min")) else 9999)
        return results


    def find_routes_with_interchange(
        self,
        origin: str,
        destination: str,
        max_changes: int = 1,
        top_n: int = 5,
    ) -> list[dict]:
        """
        Find routes using simple graph shortest-path search.
        Handles 0 or 1 interchange (train change).

        Uses NetworkX shortest simple paths weighted by travel_min.
        Each path is validated to use real train connections.

        Returns top_n routes ranked by estimated total travel time.
        """
        self._check_built()
        origin, destination = origin.upper(), destination.upper()

        # First check direct routes
        direct = self.find_direct_routes(origin, destination)
        if direct and "error" not in direct[0]:
            direct_top = direct[:top_n]
        else:
            direct_top = []

        if max_changes == 0:
            return direct_top

        # Build simple graph for path finding (one edge per station pair)
        # Use minimum travel time across all trains for that edge
        
        
        
        # simple_G = nx.DiGraph()

        # for u, v, data in self.G.edges(data=True):
        #     t = data.get("travel_min", np.nan)

        #     if pd.isna(t):
        #         t = data.get("distance_km", 60)

        #     simple_G.add_edge(
        #         u, v,
        #         weight=t,
        #         train_number=data["train_number"],
        #         train_name=data["train_name"],
        #         train_category=data["train_category"],
        #         running_days=data["running_days"],
        #         depart_time=data.get("depart_time"),
        #         arrive_time=data.get("arrive_time"),
        #         distance_km=data.get("distance_km", 0),
        #     )

        # simple_G = nx.DiGraph()

        # for u, v, data in self.G.edges(data=True):
        #     t = data.get("travel_min", np.nan)

        #     if pd.isna(t):
        #         t = data.get("distance_km", 60)

        #     if simple_G.has_edge(u, v):
        #         if t < simple_G[u][v]["weight"]:
        #             simple_G[u][v].update({
        #                 "weight": t,
        #                 "train_number": data["train_number"],
        #                 "train_name": data["train_name"],
        #                 "train_category": data["train_category"],
        #                 "running_days": data["running_days"],
        #                 "depart_time": data.get("depart_time"),
        #                 "arrive_time": data.get("arrive_time"),
        #                 "distance_km": data.get("distance_km", 0),
        #             })
        #     else:
        #         simple_G.add_edge(
        #             u, v,
        #             weight=t,
        #             train_number=data["train_number"],
        #             train_name=data["train_name"],
        #             train_category=data["train_category"],
        #             running_days=data["running_days"],
        #             depart_time=data.get("depart_time"),
        #             arrive_time=data.get("arrive_time"),
        #             distance_km=data.get("distance_km", 0),
        #         )


        # if origin not in simple_G or destination not in simple_G:
        #     return direct_top

        # # Find shortest paths (limit to avoid explosion)
        # results = list(direct_top)

        # try:
            # paths = islice(
            #     nx.shortest_simple_paths(simple_G, origin, destination, weight="weight"),
            #     top_n * 10
            # )

            # for path in paths:
                
            #     if path[-1] != destination:
            #         continue
            #     if len(results) >= top_n:
            #         break

            #     changes = 0
            #     legs    = []
            #     total_min  = 0
            #     total_dist = 0
            #     prev_train = None
            #     valid      = True

        #         for i in range(len(path) - 1):
        #             u, v  = path[i], path[i + 1]
        #             if not simple_G.has_edge(u, v):
        #                 valid = False
        #                 break
        #             edata = simple_G[u][v]
        #             t = edata["weight"]
        #             total_min  += t
        #             total_dist += edata.get("distance_km", 0)

        #             if prev_train and edata["train_number"] != prev_train:
        #                 changes += 1

        #             if changes > max_changes +1:
        #                 valid = False
        #                 break

        #             legs.append({
        #                 "leg"          : i + 1,
        #                 "from"         : u,
        #                 "from_name"    : self.get_station_name(u),
        #                 "to"           : v,
        #                 "to_name"      : self.get_station_name(v),
        #                 "train_number" : edata["train_number"],
        #                 "train_name"   : edata["train_name"],
        #                 "train_category": edata["train_category"],
        #                 "running_days" : edata["running_days"],
        #                 "depart_time"  : edata.get("depart_time"),
        #                 "arrive_time"  : edata.get("arrive_time"),
        #                 "travel_min"   : t,
        #                 "distance_km"  : edata.get("distance_km", 0),
        #             })
        #             prev_train = edata["train_number"]

        #         if not legs:
        #             continue

        #         # Skip if already captured as direct
        #         first_train = legs[0]["train_number"]
        #         if changes == 0 and any(
        #             r.get("train_number") == first_train and r["changes"] == 0
        #             for r in results
        #         ):
        #             continue

        #         delay_info = self.get_station_delay_info(origin)

        #         results.append({
        #             "route_type"         : "Direct" if changes == 0 else f"{changes}-Change",
        #             "changes"            : changes,
        #             "origin"             : origin,
        #             "origin_name"        : self.get_station_name(origin),
        #             "destination"        : destination,
        #             "destination_name"   : self.get_station_name(destination),
        #             "total_travel_min"   : round(total_min, 1),
        #             "total_travel_hrs"   : round(total_min / 60, 2),
        #             "total_distance_km"  : round(total_dist, 1),
        #             "origin_delay_min"   : delay_info["avg_delay_min"],
        #             "origin_risk"        : delay_info["risk_label"],
        #             "legs"               : legs,
        #             "train_number"       : legs[0]["train_number"] if changes == 0 else None,
        #             "train_name"         : legs[0]["train_name"]   if changes == 0 else "Multi-train",
        #             "train_category"     : legs[0]["train_category"] if changes == 0 else "Mixed",
        #             "running_days"       : legs[0]["running_days"]  if changes == 0 else "Varies",
        #         })

        # except nx.NetworkXNoPath:
        #     pass
        # except Exception as e:
        #     print(f"⚠️  Path search error: {e}")

        





        # 🔥 ADD THIS
        direct = self.find_direct_routes(origin, destination)

        if direct and "error" not in direct[0]:
            results = direct
        else:
            results = []


        # Final sort
        results.sort(key=lambda x: (x["changes"], x.get("total_travel_min") or 9999))

        return results[:top_n]

        # Final sort: fewer changes first, then by travel time
        # results.sort(key=lambda x: (x["changes"], x.get("total_travel_min") or 9999))

        # ✅ REMOVE DUPLICATE ROUTES
        unique = []
        seen = set()

        for r in results:
            key = tuple((leg["from"], leg["to"], leg["train_number"]) for leg in r["legs"])

            if key not in seen:
                seen.add(key)
                unique.append(r)

        results = unique
        
        return results[:top_n]


    def smart_route_search(
        self,
        origin: str,
        destination: str,
        top_n: int = 3,
        max_changes: int = 1,
    ) -> dict:
        """
        Main entry point for RouteMATE_AI.

        Returns a structured result dict with:
          - query info
          - top N ranked routes
          - station delay risk info
          - ready for ML delay injection
        """
        self._check_built()
        origin, destination = origin.upper(), destination.upper()
        
        import networkx as nx

        print("KOAA exists:", "KOAA" in self.G.nodes)
        print("GHY exists:", "GHY" in self.G.nodes)

        try:
            print("Has path KOAA → GHY:", nx.has_path(self.G, "KOAA", "GHY"))
        except Exception as e:
            print("Path check error:", e)

        print(f"\n🔍 Searching routes: {origin} → {destination}")

        routes = self.find_routes_with_interchange(
            origin, destination,
            max_changes=max_changes,
            top_n=top_n,
        )


        
        # # ✅ CORRECT FILTER (SINGLE + SAFE)
        # filtered_routes = []

        # for r in routes:
        #     if r.get("legs"):
        #         last_stop = r["legs"][-1].get("to")

        #         if last_stop and last_stop.upper() == destination:
        #             filtered_routes.append(r)

        # routes = filtered_routes
        # # 🔥 STRICT DESTINATION MATCH (FINAL FIX)
        # filtered_routes = []

        # for r in routes:
        #     if r.get("legs"):
        #         last_stop = r["legs"][-1].get("to_station")

        #         if last_stop == destination:
        #             filtered_routes.append(r)

        # routes = filtered_routes

        # # routes = [
        # #     r for r in routes
        # #     if r.get("legs") and r["legs"][-1].get("to_station") == destination
        # # ]

        # filtered_routes = []

        # for r in routes:
        #     if r.get("legs"):
        #         last_stop = r["legs"][-1].get("to")   # ✅ CORRECT KEY

        #         if last_stop == destination:
        #             filtered_routes.append(r)

        # routes = filtered_routes
        origin_info = self.get_station_delay_info(origin)
        dest_info   = self.get_station_delay_info(destination)

        return {
            "query": {
                "origin"         : origin,
                "origin_name"    : self.get_station_name(origin),
                "destination"    : destination,
                "destination_name": self.get_station_name(destination),
                "top_n_requested": top_n,
                "max_changes"    : max_changes,
            },
            "origin_station_info"     : origin_info,
            "destination_station_info": dest_info,
            "routes_found"            : len(routes),
            "routes"                  : routes,
        }


    def print_results(self, result: dict):
        """Pretty-print smart_route_search results to console."""
        q = result["query"]
        print(f"\n{'='*60}")
        print(f"  🚂 RouteMATE_AI Route Results")
        print(f"{'='*60}")
        print(f"  From : {q['origin_name']} ({q['origin']})")
        print(f"  To   : {q['destination_name']} ({q['destination']})")
        print(f"  Found: {result['routes_found']} route(s)")

        o = result["origin_station_info"]
        d = result["destination_station_info"]
        print(f"\n  📍 Origin delay risk   : {o['risk_label']}  (avg {o['avg_delay_min']} min)")
        print(f"  📍 Destination delay risk: {d['risk_label']}  (avg {d['avg_delay_min']} min)")

        for i, route in enumerate(result["routes"], 1):
            print(f"\n  ── Route #{i} {'─'*40}")
            print(f"     Type    : {route['route_type']}")
            print(f"     Changes : {route['changes']}")
            if route.get("train_number"):
                print(f"     Train   : {route['train_number']} — {route['train_name']}")
                print(f"     Category: {route['train_category']}")
                print(f"     Runs on : {route['running_days']}")
            hrs = route.get('total_travel_hrs') or route.get('total_travel_min', 0) / 60
            print(f"     Journey : {hrs:.1f} hrs  ({route.get('total_travel_min', '?')} min)")
            if route.get("total_distance_km"):
                print(f"     Distance: {route['total_distance_km']} km")

            print(f"     Legs:")
            for leg in route.get("legs", []):
                print(f"       Leg {leg['leg']}: {leg.get('from_name', leg['from'])} "
                      f"→ {leg.get('to_name', leg['to'])}  "
                      f"[Train {leg['train_number']}]  "
                      f"{leg.get('depart_time','?')} → {leg.get('arrive_time','?')}")
        print(f"\n{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST  (python src/graph/route_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    graph = RouteGraph()
    graph.build()

    # Test 1: Station info
    print("\n📊 Station delay info for KOAA:")
    print(graph.get_station_delay_info("KOAA"))

    # Test 2: Direct search
    result = graph.smart_route_search("KOAA", "GHY", top_n=3)
    graph.print_results(result)

    # Test 3: Another pair
    result2 = graph.smart_route_search("NDAE", "AZ", top_n=3)
    graph.print_results(result2)