"""
=============================================================
  RouteMATE_AI — Map Visualizer
  File: src/visualization/map_visualizer.py
=============================================================
  Renders interactive train route maps using Folium.
  Falls back to a clean static HTML map when Folium is
  not installed.

  Features:
    ✅ Route polylines with colour-coded risk levels
    ✅ Station markers with delay info popups
    ✅ Origin / Destination pin markers
    ✅ Interchange markers for multi-train routes
    ✅ Delay risk heatmap overlay
    ✅ Route comparison (up to 3 routes, different colours)
    ✅ Legend with risk colour key
    ✅ Saves to .html (embeddable in Streamlit via components)

  Usage:
    from src.visualization.map_visualizer import RouteMapVisualizer

    viz = RouteMapVisualizer()

    # Single route
    m = viz.render_route(route_result["routes"][0])
    viz.save(m, "route_map.html")

    # Compare top 3 routes
    m = viz.render_comparison(route_result["routes"])
    viz.save(m, "comparison_map.html")

    # In Streamlit
    viz.show_in_streamlit(m)

  Requirements:
    pip install folium
=============================================================
"""

import os
import math
import warnings
import datetime
warnings.filterwarnings("ignore")

try:
    import folium
    from folium import plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("⚠️  Folium not installed. Run: pip install folium")
    print("   Falling back to static HTML map.")

from src.utils.constants import STATION_COORDINATES, RISK_LABELS


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════

# Route colours for comparison (up to 3 routes)
ROUTE_COLORS = ["#2980b9", "#e67e22", "#27ae60"]
ROUTE_LABELS = ["Route 1 (Best)", "Route 2", "Route 3"]

# Risk → marker colour
RISK_COLORS = {
    "🟢 Low"   : "green",
    "🟡 Medium": "orange",
    "🔴 High"  : "red",
    "Unknown"  : "gray",
}

RISK_HEX = {
    "🟢 Low"   : "#27ae60",
    "🟡 Medium": "#f39c12",
    "🔴 High"  : "#e74c3c",
    "Unknown"  : "#95a5a6",
}

# Train category → icon
TRAIN_ICONS = {
    "Express"   : "train",
    "Superfast" : "forward",
    "Passenger" : "slow",
    "Rajdhani"  : "star",
}

# India map centre + default zoom
INDIA_CENTER = [22.5, 82.5]
INDIA_ZOOM   = 5


# ══════════════════════════════════════════════════════════════════════════════
#  COORDINATE RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

def get_coords(station_code: str) -> tuple | None:
    """
    Return (lat, lon) for a station code.
    Returns None if station not in coordinate database.
    """
    entry = STATION_COORDINATES.get(station_code.upper())
    if entry:
        return entry[0], entry[1]
    return None


def get_city_name(station_code: str) -> str:
    entry = STATION_COORDINATES.get(station_code.upper())
    return entry[2] if entry else station_code


def midpoint(coords_list: list) -> tuple:
    """Calculate geographic midpoint for auto-centering the map."""
    lats = [c[0] for c in coords_list if c]
    lons = [c[1] for c in coords_list if c]
    if not lats:
        return INDIA_CENTER
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def smart_zoom(coords_list: list) -> int:
    """Calculate appropriate zoom based on geographic spread."""
    lats = [c[0] for c in coords_list if c]
    lons = [c[1] for c in coords_list if c]
    if len(lats) < 2:
        return 7
    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    span = max(lat_span, lon_span)
    if span < 2:   return 9
    if span < 5:   return 7
    if span < 10:  return 6
    if span < 20:  return 5
    return 4


# ══════════════════════════════════════════════════════════════════════════════
#  POPUP BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _station_popup_html(
    code        : str,
    name        : str,
    delay_info  : dict = None,
    is_origin   : bool = False,
    is_dest     : bool = False,
    train_info  : dict = None,
) -> str:
    """Build rich HTML popup for a station marker."""
    badge = ""
    if is_origin: badge = '<span style="background:#27ae60;color:white;padding:2px 8px;border-radius:4px;font-size:11px">🚉 ORIGIN</span>'
    if is_dest:   badge = '<span style="background:#e74c3c;color:white;padding:2px 8px;border-radius:4px;font-size:11px">🏁 DESTINATION</span>'

    delay_html = ""
    if delay_info:
        risk  = delay_info.get("risk_label", "Unknown")
        color = RISK_HEX.get(risk, "#95a5a6")
        avg   = delay_info.get("avg_delay_min", "N/A")
        rt    = delay_info.get("pct_right_time", "N/A")
        cat   = delay_info.get("delay_category", "Unknown")
        delay_html = f"""
        <hr style="margin:6px 0">
        <b>Delay History</b><br>
        <span style="color:{color};font-weight:bold">{risk}</span><br>
        Avg delay: <b>{avg} min</b><br>
        On-time rate: <b>{rt}%</b><br>
        Category: <i>{cat}</i>
        """

    train_html = ""
    if train_info:
        train_html = f"""
        <hr style="margin:6px 0">
        <b>Train</b>: {train_info.get('train_number')} — {train_info.get('train_name','')}<br>
        <b>Type</b>: {train_info.get('train_category','')}<br>
        <b>Runs</b>: {train_info.get('running_days','')}
        """

    return f"""
    <div style="font-family:Arial,sans-serif;font-size:13px;min-width:180px;max-width:240px">
        {badge}
        <h4 style="margin:6px 0 4px 0;color:#2c3e50">{name}</h4>
        <span style="color:#7f8c8d;font-size:11px">Code: {code}</span>
        {delay_html}
        {train_html}
    </div>
    """


def _route_popup_html(route: dict) -> str:
    """Popup shown on clicking a route polyline."""
    rtype  = route.get("route_type", "Route")
    train  = route.get("train_name", "Multi-train")
    cat    = route.get("train_category", "")
    hrs    = route.get("total_travel_hrs", 0) or 0
    dist   = route.get("total_distance_km", "N/A")
    chg    = route.get("changes", 0)
    score  = route.get("route_score", "—")
    risk   = route.get("overall_risk", route.get("origin_risk", ""))
    days   = route.get("running_days", "")
    delay  = route.get("predicted_avg_delay_min", route.get("origin_delay_min", "N/A"))

    return f"""
    <div style="font-family:Arial,sans-serif;font-size:13px;min-width:200px">
        <h4 style="margin:0 0 6px 0;color:#2c3e50">{rtype}</h4>
        <b>{train}</b> ({cat})<br>
        ⏱ Journey: <b>{hrs:.1f} hrs</b><br>
        📏 Distance: <b>{dist} km</b><br>
        🔄 Changes: <b>{chg}</b><br>
        📅 Runs: {days}<br>
        🔮 Pred. delay: <b>{delay} min</b><br>
        ⚠️ Risk: {risk}<br>
        {"⭐ Score: <b>" + str(score) + "</b>" if score != "—" else ""}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN VISUALIZER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class RouteMapVisualizer:
    """
    Renders interactive Folium maps for RouteMATE_AI route results.
    """

    def __init__(self, tile_style: str = "CartoDB positron"):
        """
        Args:
            tile_style: Folium tile layer.
                Options: 'CartoDB positron' (clean, light — default),
                         'OpenStreetMap', 'CartoDB dark_matter'
        """
        self.tile_style   = tile_style
        self.delay_lookup = self._load_delay_lookup()

    def _load_delay_lookup(self) -> dict:
        """Load station delay info from model.pkl if available."""
        try:
            import pickle
            from src.utils.constants import MODEL_PKL
            path = MODEL_PKL
            if not os.path.exists(path):
                # Dev fallback
                path = "/home/claude/src/models/model.pkl"
            if os.path.exists(path):
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                return bundle.get("station_delay_lookup", {})
        except Exception:
            pass
        return {}

    def _get_delay_info(self, station_code: str) -> dict:
        """Get delay risk info for a station."""
        raw  = self.delay_lookup.get(station_code.upper(), {})
        avg  = raw.get("avg_delay_min", None)
        cat  = raw.get("delay_category", "Unknown")
        rt   = raw.get("pct_right_time", None)

        if avg is None:
            risk = "Unknown"
        elif avg <= 20:
            risk = "🟢 Low"
        elif avg <= 60:
            risk = "🟡 Medium"
        else:
            risk = "🔴 High"

        return {
            "avg_delay_min" : avg,
            "delay_category": cat,
            "pct_right_time": rt,
            "risk_label"    : risk,
        }

    def _base_map(self, center=None, zoom=None) -> "folium.Map":
        """Create a base Folium map."""
        return folium.Map(
            location=center or INDIA_CENTER,
            zoom_start=zoom or INDIA_ZOOM,
            tiles="CartoDB positron",   # 🔥 CLEAN WHITE MAP
            prefer_canvas=True,
        )

    # ── Station marker ────────────────────────────────────────────────────────

    def _add_station_marker(
        self,
        fmap,
        code       : str,
        name       : str,
        coords     : tuple,
        is_origin  : bool = False,
        is_dest    : bool = False,
        is_interchange: bool = False,
        train_info : dict = None,
        marker_color: str = None,
    ):
        """Add a styled station marker with popup to the map."""
        delay_info = self._get_delay_info(code)

        if marker_color is None:
            if is_origin:
                marker_color = "green"
            elif is_dest:
                marker_color = "red"
            elif is_interchange:
                marker_color = "purple"
            else:
                marker_color = RISK_COLORS.get(delay_info["risk_label"], "blue")

        icon_name = "home" if is_origin else ("flag" if is_dest else ("exchange" if is_interchange else "train"))

        popup_html = _station_popup_html(
            code, name, delay_info, is_origin, is_dest, train_info
        )

        folium.Marker(
            location = coords,
            popup    = folium.Popup(popup_html, max_width=260),
            tooltip  = f"{name} ({code})" + (
                f" — Avg delay: {delay_info['avg_delay_min']} min" if delay_info["avg_delay_min"] else ""
            ),
            icon     = folium.Icon(
                color     = marker_color,
                icon      = icon_name,
                prefix    = "fa",
            ),
        ).add_to(fmap)

    # ── Route polyline ────────────────────────────────────────────────────────

    def _add_route_line(
        self,
        fmap,
        coords     : list,
        route      : dict,
        color      : str  = "#2980b9",
        weight     : int  = 5,
        opacity    : float= 0.8,
    ):
        """Draw the route path on the map."""
        if len(coords) < 2:
            return

        folium.PolyLine(
            locations = coords,
            color     = color,
            weight    = weight,
            opacity   = opacity,
            popup     = folium.Popup(_route_popup_html(route), max_width=260),
            tooltip   = (
                f"{route.get('train_name','Route')} — "
                f"{route.get('total_travel_hrs', 0):.1f} hrs"
            ),
            dash_array= None if route.get("changes", 0) == 0 else "10",
        ).add_to(fmap)

        # Animated dashed line overlay for multi-train routes
        if route.get("changes", 0) > 0:
            plugins.AntPath(
                locations = coords,
                color     = color,
                weight    = 3,
                opacity   = 0.5,
                delay     = 800,
            ).add_to(fmap)

    # ── Legend ────────────────────────────────────────────────────────────────

    def _add_legend(self, fmap, routes: list):
        """Add a colour legend to the map."""
        items = []
        for i, route in enumerate(routes[:3]):
            color = ROUTE_COLORS[i]
            label = route.get("train_name", f"Route {i+1}")
            hrs   = route.get("total_travel_hrs", 0) or 0
            items.append(
                f'<li><span style="background:{color};width:20px;height:4px;'
                f'display:inline-block;vertical-align:middle;margin-right:8px"></span>'
                f'{label} ({hrs:.1f}h)</li>'
            )

        risk_items = "".join([
            f'<li><span style="background:{RISK_HEX[r]};width:12px;height:12px;'
            f'border-radius:50%;display:inline-block;vertical-align:middle;margin-right:6px"></span>'
            f'{r}</li>'
            for r in ["🟢 Low", "🟡 Medium", "🔴 High"]
        ])

        legend_html = f"""
        <div style="
            position: fixed; bottom: 30px; left: 30px; z-index: 1000;
            background: white; padding: 14px 18px; border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif; font-size: 13px;
            min-width: 200px;
        ">
            <b style="color:#2c3e50">🚂 RouteMATE AI</b>
            <ul style="list-style:none;padding:0;margin:8px 0 4px 0">
                {"".join(items)}
            </ul>
            <hr style="margin:6px 0;border-color:#eee">
            <b style="font-size:11px;color:#7f8c8d">DELAY RISK</b>
            <ul style="list-style:none;padding:0;margin:4px 0 0 0;font-size:12px">
                {risk_items}
            </ul>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))

    # ══════════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def render_route(
        self,
        route        : dict,
        show_waypoints: bool = True,
    ) -> "folium.Map | str":
        """
        Render a single route on an interactive map.

        Args:
            route        : single route dict from smart_route_search()
            show_waypoints: show intermediate station markers

        Returns:
            folium.Map object (or HTML string in fallback mode)
        """
        if not FOLIUM_AVAILABLE:
            return self._fallback_html([route])

        legs   = route.get("legs", [])
        origin = route.get("origin", "")
        dest   = route.get("destination", "")

        # Collect all station coords for this route
        all_codes  = [origin] + [l.get("to", "") for l in legs]
        all_coords = [get_coords(c) for c in all_codes]
        all_coords = [c for c in all_coords if c]

        if not all_coords:
            # No coords known — return fallback
            return self._fallback_html([route])

        # Create map centred on route
        center = midpoint(all_coords)
        zoom   = smart_zoom(all_coords)
        fmap   = self._base_map(center, zoom)

        # Draw route line
        self._add_route_line(fmap, all_coords, route, color=ROUTE_COLORS[0])

        # Add origin marker
        o_coords = get_coords(origin)
        if o_coords:
            self._add_station_marker(
                fmap, origin, get_city_name(origin),
                o_coords, is_origin=True,
                train_info=legs[0] if legs else None,
            )

        # Add destination marker
        d_coords = get_coords(dest)
        if d_coords:
            self._add_station_marker(
                fmap, dest, get_city_name(dest),
                d_coords, is_dest=True,
            )

        # Add intermediate waypoints
        if show_waypoints:
            for leg in legs[:-1]:
                via_code   = leg.get("to", "")
                via_coords = get_coords(via_code)
                if via_coords and via_code not in (origin, dest):
                    is_chg = len(legs) > 1
                    self._add_station_marker(
                        fmap, via_code, get_city_name(via_code),
                        via_coords,
                        is_interchange=is_chg,
                        train_info=leg,
                    )

        # Add all known delay stations as small circles
        self._add_delay_heatpoints(fmap, exclude={origin, dest})

        # Legend
        self._add_legend(fmap, [route])

        return fmap

    def render_comparison(
        self,
        routes       : list,
        show_all_stations: bool = True,
    ) -> "folium.Map | str":
        """
        Render up to 3 routes on the same map with different colours.
        Perfect for the Streamlit 'Top 3 Routes' panel.

        Args:
            routes: list of route dicts (from smart_route_search)

        Returns:
            folium.Map object
        """
        if not FOLIUM_AVAILABLE:
            return self._fallback_html(routes)

        if not routes:
            return self._base_map()

        # Collect all coords across all routes
        all_coords_flat = []
        for route in routes:
            legs = route.get("legs", [])
            codes = [route.get("origin", "")] + [l.get("to", "") for l in legs]
            for c in codes:
                coords = get_coords(c)
                if coords:
                    all_coords_flat.append(coords)

        center = midpoint(all_coords_flat) if all_coords_flat else INDIA_CENTER
        zoom   = smart_zoom(all_coords_flat) if all_coords_flat else INDIA_ZOOM
        fmap   = self._base_map(center, zoom)

        origin = routes[0].get("origin", "")
        dest   = routes[0].get("destination", "")

        # Draw each route
        for i, route in enumerate(routes[:3]):
            color  = ROUTE_COLORS[i]
            legs   = route.get("legs", [])
            codes  = [route.get("origin", "")] + [l.get("to", "") for l in legs]
            coords = [get_coords(c) for c in codes if get_coords(c)]

            if coords:
                self._add_route_line(
                    fmap, coords, route,
                    color=color,
                    weight=5 - i,        # Route 1 thickest
                    opacity=0.9 - i*0.1,
                )

        # Origin and destination pins (shared across routes)
        o_coords = get_coords(origin)
        d_coords = get_coords(dest)
        if o_coords:
            self._add_station_marker(fmap, origin, get_city_name(origin),
                                     o_coords, is_origin=True)
        if d_coords:
            self._add_station_marker(fmap, dest, get_city_name(dest),
                                     d_coords, is_dest=True)

        # Delay risk markers for all known stations
        if show_all_stations:
            self._add_delay_heatpoints(fmap, exclude={origin, dest})

        # Legend
        self._add_legend(fmap, routes[:3])

        return fmap

    def render_network(self) -> "folium.Map | str":
        """
        Render all stations with known delay data as a risk heatmap.
        Useful as a standalone India-wide delay overview map.
        """
        if not FOLIUM_AVAILABLE:
            return self._fallback_html([])

        fmap = self._base_map(INDIA_CENTER, INDIA_ZOOM)
        self._add_delay_heatpoints(fmap, exclude=set(), show_all=True)

        # Title box
        title_html = """
        <div style="
            position:fixed; top:15px; left:50%; transform:translateX(-50%);
            background:white; padding:10px 20px; border-radius:8px;
            box-shadow:0 2px 8px rgba(0,0,0,0.2);
            font-family:Arial; font-size:15px; font-weight:bold; color:#2c3e50;
            z-index:999;
        ">
            🚂 RouteMATE AI — Station Delay Risk Network
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(title_html))
        self._add_legend(fmap, [])
        return fmap

    def _add_delay_heatpoints(self, fmap, exclude: set, show_all: bool = False):
        """Add small circle markers for all stations with known delay data."""
        for code, info in self.delay_lookup.items():
            if code in exclude:
                continue
            coords = get_coords(code)
            if not coords:
                continue

            avg  = info.get("avg_delay_min", 0) or 0
            risk = ("🟢 Low" if avg <= 20 else "🟡 Medium" if avg <= 60 else "🔴 High")
            color = RISK_HEX.get(risk, "#95a5a6")

            folium.CircleMarker(
                location     = coords,
                radius       = max(5, min(avg / 15, 18)),
                color        = color,
                fill         = True,
                fill_color   = color,
                fill_opacity = 0.55,
                popup        = folium.Popup(
                    _station_popup_html(code, get_city_name(code), {
                        "avg_delay_min" : avg,
                        "delay_category": info.get("delay_category",""),
                        "pct_right_time": info.get("pct_right_time",""),
                        "risk_label"    : risk,
                    }),
                    max_width=240,
                ),
                tooltip=f"{get_city_name(code)} ({code}) — {avg} min avg delay",
            ).add_to(fmap)

    # ── Save / Show ───────────────────────────────────────────────────────────

    def save(self, fmap, path: str = "route_map.html"):
        """Save map to HTML file."""
        if isinstance(fmap, str):
            # Fallback HTML string
            with open(path, "w", encoding="utf-8") as f:
                f.write(fmap)
        else:
            fmap.save(path)
        print(f"🗺️  Map saved → {path}")

    def show_in_streamlit(self, fmap, height: int = 500):
        """
        Display map inside Streamlit app.

        Usage in app.py:
            import streamlit.components.v1 as components
            viz = RouteMapVisualizer()
            m   = viz.render_comparison(routes)
            viz.show_in_streamlit(m)
        """
        try:
            import streamlit.components.v1 as components
            if isinstance(fmap, str):
                html_str = fmap
            else:
                html_str = fmap._repr_html_()
            components.html(html_str, height=height, scrolling=False)
        except ImportError:
            print("⚠️  Streamlit not available. Use .save() to export HTML.")

    def get_html_string(self, fmap) -> str:
        """Return the map as an HTML string (for Streamlit components.html)."""
        if isinstance(fmap, str):
            return fmap
        return fmap._repr_html_()

    # ══════════════════════════════════════════════════════════════════════════
    #  STATIC HTML FALLBACK (when Folium not installed)
    # ══════════════════════════════════════════════════════════════════════════

    def _fallback_html(self, routes: list) -> str:
        """
        Generate a clean static HTML card when Folium is unavailable.
        Shows route info in a formatted card — embeddable in Streamlit.
        """
        cards = ""
        for i, route in enumerate(routes[:3]):
            color   = ROUTE_COLORS[i]
            rtype   = route.get("route_type", "Route")
            train   = route.get("train_name", "—")
            hrs     = route.get("total_travel_hrs", 0) or 0
            dist    = route.get("total_distance_km", "N/A")
            chg     = route.get("changes", 0)
            risk    = route.get("overall_risk", route.get("origin_risk", ""))
            score   = route.get("route_score", "—")
            origin  = route.get("origin_name", route.get("origin", ""))
            dest    = route.get("destination_name", route.get("destination", ""))

            cards += f"""
            <div style="border-left:5px solid {color};padding:12px 16px;margin-bottom:14px;
                        background:#f8f9fa;border-radius:6px">
                <div style="font-weight:bold;font-size:15px;color:{color}">
                    Route {i+1} — {rtype}
                </div>
                <div style="margin-top:6px;font-size:13px;color:#2c3e50">
                    🚂 {train}<br>
                    📍 {origin} → {dest}<br>
                    ⏱ {hrs:.1f} hrs &nbsp;|&nbsp; 📏 {dist} km
                    &nbsp;|&nbsp; 🔄 {chg} change(s)<br>
                    ⚠️ Risk: {risk} &nbsp;|&nbsp; ⭐ Score: {score}
                </div>
            </div>
            """

        install_note = "" if FOLIUM_AVAILABLE else """
            <div style="background:#fff3cd;padding:10px 14px;border-radius:6px;
                        font-size:12px;color:#856404;margin-bottom:14px">
                📦 Install <code>folium</code> for an interactive map:
                <code>pip install folium</code>
            </div>
        """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: white;
                    padding: 16px;
                    margin: 0;
                }}
            </style>
        </head>
        <body>
            <div style="font-size:18px;font-weight:bold;color:#2c3e50;margin-bottom:12px">
                🗺️ RouteMATE AI — Route Overview
            </div>
            {install_note}
            {cards if cards else '<p style="color:#7f8c8d">No routes to display.</p>'}
        </body>
        </html>
        """


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    print("=" * 55)
    print("  MapVisualizer — Test")
    print("=" * 55)
    print(f"  Folium available: {FOLIUM_AVAILABLE}")

    # Build a mock route (same structure as route_engine output)
    mock_route_1 = {
        "route_type"         : "Direct",
        "changes"            : 0,
        "train_number"       : 13125,
        "train_name"         : "Koaa Sairang Exp",
        "train_category"     : "Express",
        "running_days"       : "TUE,WED,SAT",
        "origin"             : "KOAA",
        "origin_name"        : "Kolkata",
        "destination"        : "GHY",
        "destination_name"   : "Guwahati",
        "total_travel_min"   : 1085,
        "total_travel_hrs"   : 18.1,
        "total_distance_km"  : 1144,
        "origin_delay_min"   : 7,
        "origin_risk"        : "🟢 Low",
        "route_score"        : 72,
        "overall_risk"       : "🔴 High",
        "legs"               : [
            {"leg":1, "from":"KOAA","from_name":"Kolkata",
             "to":"NJP", "to_name":"New Jalpaiguri",
             "train_number":13125, "train_name":"Koaa Sairang Exp",
             "train_category":"Express","running_days":"TUE,WED,SAT",
             "depart_time":"06:00","arrive_time":"14:30",
             "travel_min":510, "distance_km":570},
            {"leg":2, "from":"NJP","from_name":"New Jalpaiguri",
             "to":"GHY", "to_name":"Guwahati",
             "train_number":13125, "train_name":"Koaa Sairang Exp",
             "train_category":"Express","running_days":"TUE,WED,SAT",
             "depart_time":"14:45","arrive_time":"23:00",
             "travel_min":575, "distance_km":574},
        ],
    }

    mock_route_2 = {**mock_route_1,
        "train_number": 12525, "train_name": "Koaa Dbrg Exp",
        "total_travel_hrs": 18.2, "route_score": 68,
    }

    viz = RouteMapVisualizer()

    # Single route map
    m1 = viz.render_route(mock_route_1)
    viz.save(m1, "/mnt/user-data/outputs/route_map_single.html")

    # Comparison map
    m2 = viz.render_comparison([mock_route_1, mock_route_2])
    viz.save(m2, "/mnt/user-data/outputs/route_map_comparison.html")

    # Network overview
    m3 = viz.render_network()
    viz.save(m3, "/mnt/user-data/outputs/route_map_network.html")

    print("\n  ✅ All maps saved:")
    print("     route_map_single.html")
    print("     route_map_comparison.html")
    print("     route_map_network.html")
    print("\n  📌 To use in app.py:")
    print("     viz = RouteMapVisualizer()")
    print("     m   = viz.render_comparison(result['routes'])")
    print("     viz.show_in_streamlit(m, height=520)")