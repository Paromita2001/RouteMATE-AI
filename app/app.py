"""
=============================================================
  RouteMATE_AI — Streamlit App
  File: app/app.py
=============================================================
  Run: streamlit run app/app.py
=============================================================
"""

import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title = "RouteMATE AI",
    page_icon  = "🚂",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 1rem;
    opacity: 0.75;
    margin: 0.3rem 0 0 0;
}

/* ── Route Card ── */
.route-card {
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #e8ecf0;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.15s;
}
.route-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.route-card.best {
    border-left: 5px solid #2980b9;
    background: linear-gradient(to right, #f0f7ff, white);
}
.route-card.second { border-left: 5px solid #e67e22; }
.route-card.third  { border-left: 5px solid #27ae60; }

/* ── Metric pill ── */
.metric-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 3px;
}
.pill-blue   { background:#dbeafe; color:#1d4ed8; }
.pill-green  { background:#dcfce7; color:#15803d; }
.pill-orange { background:#ffedd5; color:#c2410c; }
.pill-red    { background:#fee2e2; color:#b91c1c; }
.pill-purple { background:#f3e8ff; color:#7c3aed; }

/* ── Score badge ── */
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
}
.score-excellent { background:#d1fae5; color:#065f46; }
.score-good      { background:#dbeafe; color:#1e40af; }
.score-fair      { background:#fef3c7; color:#92400e; }
.score-risk      { background:#fee2e2; color:#991b1b; }

/* ── Stat box ── */
.stat-box {
    background: #f8fafc;
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    border: 1px solid #e2e8f0;
}
.stat-box .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1;
}
.stat-box .label {
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Section header ── */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin: 1.2rem 0 0.8rem 0;
}

/* ── Context banner ── */
.context-banner {
    background: #f1f5f9;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 13px;
    color: #334155;
    margin-bottom: 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stDateInput label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label {
    color: #94a3b8 !important;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── Hide streamlit branding ── */
#MainMenu, footer { visibility: hidden; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* ── Alert boxes ── */
.alert-warning {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 13px;
    margin: 8px 0;
}
.alert-info {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 13px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_predictor():
    from src.models.predict import RoutePredictor
    return RoutePredictor()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def score_class(score: float) -> str:
    if score >= 80: return "score-excellent"
    if score >= 60: return "score-good"
    if score >= 40: return "score-fair"
    return "score-risk"


def risk_color(risk: str) -> str:
    if "Low"    in risk: return "#15803d"
    if "Medium" in risk: return "#b45309"
    return "#b91c1c"


def format_mins(mins) -> str:
    if mins is None: return "N/A"
    h = int(mins) // 60
    m = int(mins) % 60
    return f"{h}h {m}m" if h else f"{m}m"


def render_route_card(route: dict, rank: int):
    """Render a rich route card."""
    card_class = {1: "best", 2: "second", 3: "third"}.get(rank, "")
    color      = {1: "#2980b9", 2: "#e67e22", 3: "#27ae60"}.get(rank, "#64748b")
    medal      = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")

    train    = route.get("train_name", "Unknown")
    train_no = route.get("train_number", "")
    cat      = route.get("train_category", "")
    rtype    = route.get("route_type", "Route")
    changes  = route.get("changes", 0)
    days     = route.get("running_days", "—")
    hrs      = route.get("total_travel_hrs", 0) or 0
    dist     = route.get("total_distance_km", "—")
    delay    = route.get("adjusted_delay_min", 0) or 0
    risk     = route.get("overall_risk", "Unknown")
    score    = route.get("adjusted_score", 0)
    label    = route.get("adjusted_label", "")
    fare     = route.get("fare_estimate", {}).get("total_fare", 0)
    fare_pp  = route.get("fare_estimate", {}).get("fare_per_passenger", 0)
    rc       = score_class(score)

    pills = f"""
    <span class="metric-pill pill-blue">⏱ {hrs:.1f} hrs</span>
    <span class="metric-pill pill-purple">📏 {dist} km</span>
    <span class="metric-pill {'pill-green' if changes==0 else 'pill-orange'}">
        🔄 {'Direct' if changes==0 else f'{changes} change(s)'}
    </span>
    <span class="metric-pill pill-blue">🗓 {days}</span>
    """

    st.markdown(f"""
    <div class="route-card {card_class}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
            <div>
                <div style="font-size:18px;font-weight:700;color:#1e293b">
                    {medal} {train}
                    <span style="font-size:12px;color:#94a3b8;font-weight:400;margin-left:6px">
                        #{train_no} · {cat}
                    </span>
                </div>
                <div style="margin-top:6px">{pills}</div>
            </div>
            <div style="text-align:right">
                <div class="score-badge {rc}">{label}</div>
                <div style="font-size:22px;font-weight:700;color:{color};margin-top:4px">
                    ₹{fare:,}
                </div>
                <div style="font-size:11px;color:#94a3b8">
                    ₹{fare_pp:,}/person
                </div>
            </div>
        </div>
        <hr style="margin:10px 0">
        <div style="display:flex;gap:24px;flex-wrap:wrap;font-size:13px">
            <div>
                <span style="color:#64748b">Pred. Delay</span><br>
                <span style="font-weight:600;color:{risk_color(risk)}">
                    ~{delay:.0f} min
                </span>
            </div>
            <div>
                <span style="color:#64748b">Risk Level</span><br>
                <span style="font-weight:600">{risk}</span>
            </div>
            <div>
                <span style="color:#64748b">Route Type</span><br>
                <span style="font-weight:600">{rtype}</span>
            </div>
            <div>
                <span style="color:#64748b">Smart Score</span><br>
                <span style="font-weight:700;color:{color}">{score}/100</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_leg_timeline(route: dict):
    """Render a stop-by-stop leg timeline."""
    legs = route.get("legs", [])
    if not legs:
        st.info("No leg details available.")
        return

    for i, leg in enumerate(legs):
        fr       = leg.get("from_name", leg.get("from", "?"))
        to       = leg.get("to_name",   leg.get("to",   "?"))
        train_no = leg.get("train_number", "?")
        dep      = leg.get("depart_time", "—") or "—"
        arr      = leg.get("arrive_time", "—") or "—"
        mins     = leg.get("travel_min")
        dur      = format_mins(mins) if mins else "—"
        dist     = leg.get("distance_km", 0)
        dp       = leg.get("delay_prediction", {})
        pred_d   = dp.get("predicted_delay_min", "—")
        d_risk   = dp.get("risk_label", "")

        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.markdown(f"""
            <div style="font-size:13px">
                <div style="font-weight:600;color:#1e293b">🚉 {fr}</div>
                <div style="color:#64748b;font-size:11px">Depart: {dep}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="text-align:center;font-size:12px;color:#64748b;padding:6px">
                ──── 🚂 Train {train_no} ─── {dur} · {dist} km ────
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="font-size:13px;text-align:right">
                <div style="font-weight:600;color:#1e293b">🏁 {to}</div>
                <div style="color:#64748b;font-size:11px">Arrive: {arr}</div>
                <div style="font-size:11px">{d_risk}</div>
            </div>""", unsafe_allow_html=True)

        if pred_d != "—":
            st.markdown(
                f'<div class="alert-info" style="font-size:12px">'
                f'🔮 Predicted delay at <b>{to}</b>: <b>{pred_d:.0f} min</b> {d_risk}'
                f'</div>', unsafe_allow_html=True)

        if i < len(legs) - 1:
            st.markdown("---")


def render_context_banner(context: dict):
    """Show calendar and weather context strip."""
    if not context:
        return
    cal   = context["calendar"]
    adv   = context["travel_advisory"]
    w     = context["worst_weather"]
    wfact = context["weather_risk_factor"]
    dem   = context["demand_label"]

    w_icon = {"Clear":"☀️","Partly Cloudy":"⛅","Cloudy":"☁️","Rain":"🌧️",
               "Heavy Rain":"⛈️","Fog":"🌫️","Thunderstorm":"⛈️",
               "Snow":"❄️","Drizzle":"🌦️"}.get(w, "🌤")

    st.markdown(f"""
    <div class="context-banner">
        📅 <b>{cal['day_of_week']}, {cal['date']}</b> &nbsp;|&nbsp;
        {dem} &nbsp;|&nbsp;
        {w_icon} {w} (risk ×{wfact:.1f}) &nbsp;|&nbsp;
        {adv}
    </div>""", unsafe_allow_html=True)


def render_fare_table(fare_options: list, selected_class: str):
    """Show class-wise fare comparison."""
    if not fare_options:
        return
    rows = []
    for f in fare_options:
        rows.append({
            "Class"     : f["class"],
            "Full Name" : {
                "1A":"First AC","2A":"2nd AC","3A":"3rd AC",
                "SL":"Sleeper","CC":"Chair Car","2S":"2nd Sitting","GEN":"General"
            }.get(f["class"],"—"),
            "Fare/Person": f"₹{f['fare_per_passenger']:,}",
            "Total Fare" : f"₹{f['total_fare']:,}",
            "Tatkal"     : f"₹{f['tatkal_total']:,}",
            "₹/km"       : f["cost_per_km"],
            "Value"      : f["value_rating"],
            "Selected"   : "✅" if f["class"] == selected_class else "",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={
                     "Value"   : st.column_config.TextColumn("Value Rating"),
                     "₹/km"   : st.column_config.NumberColumn("₹/km", format="%.2f"),
                     "Selected": st.column_config.TextColumn("Your Class"),
                 })


def render_availability(avail: dict, travel_class: str):
    """Show seat availability summary."""
    if not avail:
        return

    best  = avail.get("best_class")
    summ  = avail.get("summary", "")
    by_cls= avail.get("by_class", {})

    cols = st.columns(len(by_cls) or 1)
    for i, (cls, info) in enumerate(by_cls.items()):
        with cols[i % len(cols)]:
            status = info["availability_status"]
            color  = {
                "Available": "#15803d", "Limited": "#b45309",
                "RAC": "#7c3aed", "Waitlist": "#b91c1c",
            }.get(status, "#64748b")
            highlight = "border: 2px solid #2563eb;" if cls == travel_class else ""
            st.markdown(f"""
            <div class="stat-box" style="{highlight}">
                <div style="font-size:15px;font-weight:700">{cls}</div>
                <div style="color:{color};font-weight:600;font-size:13px">{status}</div>
                <div style="font-size:11px;color:#64748b">{info['note']}</div>
            </div>""", unsafe_allow_html=True)

    if best:
        st.success(f"✅ Best available: **{best}** — {summ}")
    else:
        st.warning(f"⚠️ {summ}")


def render_map(routes: list):
    """Render Folium map or fallback."""
    try:
        from src.visualization.map_visualizer import RouteMapVisualizer
        import streamlit.components.v1 as components
        viz = RouteMapVisualizer()
        m   = viz.render_comparison(routes)
        html_str = viz.get_html_string(m)
        if html_str.strip().startswith("<!DOCTYPE"):
            # Static fallback HTML
            components.html(html_str, height=340, scrolling=False)
        else:
            components.html(html_str, height=480, scrolling=False)
    except Exception as e:
        st.info(f"🗺️ Map unavailable: {e}. Install `folium` for interactive maps.")


def render_explainability(route: dict):
    """Show SHAP / LIME explanations for the first leg."""
    try:
        from src.explainability.shap_explainer import SHAPExplainer
        from src.explainability.lime_explainer import LIMEExplainer
        from src.utils.helpers import build_feature_row
        from src.utils.constants import MODEL_PKL

        legs = route.get("legs", [])
        if not legs:
            st.info("No legs to explain.")
            return

        leg = legs[-1]  # last (destination) leg is most interesting
        dp  = leg.get("delay_prediction", {})

        row = build_feature_row(
            station_code    = leg.get("to", "KOAA"),
            train_category  = leg.get("train_category", "Express"),
            stop_position   = 0.9,
            distance_km     = leg.get("distance_km", 500),
            halt_min        = 5,
            time_of_day_min = 720,
            is_overnight    = 1 if route.get("total_travel_hrs", 0) > 12 else 0,
            journey_day     = 2 if route.get("total_travel_hrs", 0) > 20 else 1,
            running_days    = route.get("running_days", "MON"),
            total_stops     = len(legs) + 5,
            station_pct_rt  = 50.0,
            station_pct_sl  = 20.0,
            station_pct_sig = 30.0,
        )

        tab1, tab2 = st.tabs(["🔵 SHAP Explanation", "🟠 LIME Explanation"])

        with tab1:
            shap_exp = SHAPExplainer(MODEL_PKL)
            shap_exp.build()
            result = shap_exp.explain(row)
            st.markdown(f"**Method:** `{result['method']}`")
            st.markdown(f"**Predicted delay:** `{result['predicted_value']} min`  |  "
                        f"**Baseline:** `{result['base_value']} min`")
            st.text(result["explanation_text"])
            fig = shap_exp.plot_waterfall(result, title="SHAP — Delay Drivers")
            if fig:
                st.pyplot(fig)

        with tab2:
            lime_exp = LIMEExplainer(MODEL_PKL)
            lime_exp.build()
            result = lime_exp.explain(row)
            st.markdown(f"**Method:** `{result['method']}`")
            st.text(result["explanation_text"])
            fig = lime_exp.plot(result, title="LIME — Local Feature Weights")
            if fig:
                st.pyplot(fig)

    except Exception as e:
        st.warning(f"Explainability unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(predictor) -> dict:
    """Render sidebar inputs. Returns form values dict."""

    st.sidebar.markdown("""
    <div style="padding:12px 0 20px 0;text-align:center">
        <div style="font-size:28px">🚂</div>
        <div style="font-size:18px;font-weight:700;color:white">RouteMATE AI</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">Smart Railway Route Planner</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="color:#94a3b8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px">QUICK ROUTES</div>', unsafe_allow_html=True)

    from src.utils.constants import POPULAR_ROUTES
    # quick_labels = ["— Select —"] + [r[2] for r in POPULAR_ROUTES]
    # quick_sel = st.sidebar.selectbox("Quick Routes", quick_labels, label_visibility="collapsed")

    # default_origin = ""
    # default_dest   = ""
    # if quick_sel != "— Select —":
    #     for code_o, code_d, label in POPULAR_ROUTES:
    #         if label == quick_sel:
    #             default_origin = code_o
    #             default_dest   = code_d
    #             break

    # 🔥 MODE SELECTOR
    mode = st.sidebar.radio(
        "Select Mode",
        ["Quick Routes", "Manual Selection"]
    )

    default_origin = "KOAA"
    default_dest = "GHY"

    # 🔥 QUICK ROUTES MODE
    if mode == "Quick Routes":

        quick_labels = ["— Select —"] + [r[2] for r in POPULAR_ROUTES]

        quick_sel = st.sidebar.selectbox(
            "Quick Routes",
            quick_labels,
            key="quick_routes"
        )

        if quick_sel != "— Select —":
            for code_o, code_d, label in POPULAR_ROUTES:
                if label == quick_sel:
                    origin = code_o
                    destination = code_d
                    break
        else:
            origin = default_origin
            destination = default_dest

    # 🔥 MANUAL MODE (MAIN FIX)
    else:
        stations = sorted(predictor.graph.G.nodes)

        origin = st.sidebar.selectbox(
            "From Station",
            stations, 
            index=stations.index(default_origin) if default_origin in stations else 0,
            key="manual_origin"
        )

        destination = st.sidebar.selectbox(
            "To Station",
            stations,
            index=stations.index(default_dest) if default_dest in stations else 0,
            key="manual_destination"
        )
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="color:#94a3b8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">JOURNEY DETAILS</div>', unsafe_allow_html=True)

    # origin      = st.sidebar.text_input("From (Station Code)", value=default_origin,
    #                                      placeholder="e.g. KOAA").upper().strip()
    # destination = st.sidebar.text_input("To (Station Code)", value=default_dest,
    #                                      placeholder="e.g. GHY").upper().strip()

    stations = sorted(predictor.graph.G.nodes)

    origin = st.sidebar.selectbox(
        "From Station",
        stations,
        index=stations.index(default_origin) if default_origin in stations else 0
    )

    destination = st.sidebar.selectbox(
        "To Station",
        stations,
        index=stations.index(default_dest) if default_dest in stations else 0
    )
    travel_date = st.sidebar.date_input(
        "Travel Date",
        value=datetime.date.today() + datetime.timedelta(days=1),
        min_value=datetime.date.today(),
        max_value=datetime.date.today() + datetime.timedelta(days=120),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="color:#94a3b8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">PREFERENCES</div>', unsafe_allow_html=True)

    travel_class = st.sidebar.selectbox(
        "Travel Class",
        ["SL", "3A", "2A", "1A", "CC", "2S", "GEN"],
        index=0,
        format_func=lambda c: {
            "SL":"🛏 Sleeper (SL)", "3A":"❄️ 3rd AC (3A)",
            "2A":"❄️ 2nd AC (2A)", "1A":"⭐ 1st AC (1A)",
            "CC":"💺 Chair Car (CC)", "2S":"💺 2nd Sitting (2S)",
            "GEN":"🎟 General (GEN)"
        }.get(c, c)
    )
    passengers = st.sidebar.number_input("Passengers", min_value=1, max_value=6, value=1)
    top_n      = st.sidebar.slider("Routes to Show", 1, 5, 3)

    st.sidebar.markdown("---")
    search_clicked = st.sidebar.button("🔍  Search Smart Routes", use_container_width=True)

    st.sidebar.markdown("""
    <div style="margin-top:20px;padding:12px;background:#1e293b;border-radius:8px;font-size:11px;color:#64748b">
        <b style="color:#94a3b8">How it works:</b><br>
        1. Graph engine finds routes<br>
        2. ML model predicts delays<br>
        3. Weather + calendar context<br>
        4. Smart score ranks routes
    </div>
    """, unsafe_allow_html=True)

    return {
        "origin"       : origin,
        "destination"  : destination,
        "travel_date"  : travel_date,
        "travel_class" : travel_class,
        "passengers"   : passengers,
        "top_n"        : top_n,
        "search"       : search_clicked,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🚂 RouteMATE AI</h1>
        <p>Smart Indian Railway Route Planner · ML-Powered Delay Prediction · Real-Time Risk Scoring</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load predictor ────────────────────────────────────────────────────────
    with st.spinner("Loading RouteMATE AI engine..."):
        predictor = load_predictor()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    inputs = render_sidebar(predictor)

    # ── Default landing state ─────────────────────────────────────────────────
    if not inputs["search"] and "last_result" not in st.session_state:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="stat-box" style="padding:20px">
                <div style="font-size:2rem">8,607</div>
                <div style="color:#64748b;font-size:13px">Stations in Network</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="stat-box" style="padding:20px">
                <div style="font-size:2rem">8,490</div>
                <div style="color:#64748b;font-size:13px">Trains Indexed</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="stat-box" style="padding:20px">
                <div style="font-size:2rem">161K</div>
                <div style="color:#64748b;font-size:13px">Route Connections</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-info" style="margin-top:1.5rem;font-size:14px">
            👈 Enter a <b>From</b> and <b>To</b> station code in the sidebar and click
            <b>Search Smart Routes</b> to get started.<br><br>
            Try: <b>KOAA → GHY</b> (Kolkata to Guwahati) or
            <b>NDLS → BCT</b> (New Delhi to Mumbai)
        </div>
        """, unsafe_allow_html=True)

        # Show network delay map
        st.markdown('<div class="section-title">🗺️ Station Delay Risk Network</div>',
                    unsafe_allow_html=True)
        try:
            from src.visualization.map_visualizer import RouteMapVisualizer
            import streamlit.components.v1 as components
            viz = RouteMapVisualizer()
            m   = viz.render_network()
            components.html(viz.get_html_string(m), height=420, scrolling=False)
        except Exception:
            st.info("Install `folium` for interactive maps: `pip install folium`")
        return

    # ── Run prediction ────────────────────────────────────────────────────────
    if inputs["search"]:
        if not inputs["origin"] or not inputs["destination"]:
            st.error("Please enter both origin and destination station codes.")
            return

        with st.spinner(f"🔍 Finding smart routes from **{inputs['origin']}** to **{inputs['destination']}**..."):
            result = predictor.predict(
                origin       = inputs["origin"],
                destination  = inputs["destination"],
                travel_date  = inputs["travel_date"],
                travel_class = inputs["travel_class"],
                passengers   = inputs["passengers"],
                top_n        = inputs["top_n"],
                max_changes = 3
            )
        st.session_state["last_result"] = result
        st.session_state["last_inputs"] = inputs

    result = st.session_state.get("last_result")
    inputs = st.session_state.get("last_inputs", inputs)
    if not result:
        return

    # ── Validation errors ─────────────────────────────────────────────────────
    if not result["validation"]["valid"]:
        for err in result["validation"]["errors"]:
            st.error(f"❌ {err}")
        return

    if not result["routes"]:
        st.warning(f"No routes found between {inputs['origin']} and {inputs['destination']}.")
        return

    # ── Journey header ────────────────────────────────────────────────────────
    q = result["query"]
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.markdown(f"""
        <div style="font-size:20px;font-weight:700;color:#1e293b">
            {q['origin_name']} <span style="color:#94a3b8">({q['origin']})</span>
            &nbsp;→&nbsp;
            {q['destination_name']} <span style="color:#94a3b8">({q['destination']})</span>
        </div>
        <div style="font-size:13px;color:#64748b;margin-top:2px">
            {q['travel_date']} · {q['travel_class']} · {q['passengers']} pax
        </div>""", unsafe_allow_html=True)
    with c2:
        best = result["best_route"]
        st.markdown(f"""
        <div class="stat-box">
            <div class="value">{len(result['routes'])}</div>
            <div class="label">Routes Found</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        hrs = best.get("total_travel_hrs", 0) if best else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="value">{hrs:.1f}h</div>
            <div class="label">Best Journey</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        fare = best.get("fare_estimate", {}).get("total_fare", 0) if best else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="value">₹{fare:,}</div>
            <div class="label">Est. Fare</div>
        </div>""", unsafe_allow_html=True)

    # ── Context banner ────────────────────────────────────────────────────────
    render_context_banner(result.get("context"))

    # ── Tatkal warning ────────────────────────────────────────────────────────
    if result.get("tatkal_recommended"):
        st.markdown(f"""
        <div class="alert-warning">
            ⚠️ <b>Tatkal Recommended:</b> {result['tatkal_reason']}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Main tabs ─────────────────────────────────────────────────────────────
    tab_routes, tab_map, tab_fare, tab_avail, tab_explain = st.tabs([
        "🚂 Routes", "🗺️ Map", "💰 Fares", "🎟 Availability", "🔍 Explain AI"
    ])

    # ── Tab 1: Routes ─────────────────────────────────────────────────────────
    with tab_routes:
        st.markdown('<div class="section-title">Top Smart Routes</div>',
                    unsafe_allow_html=True)

        for i, route in enumerate(result["routes"], 1):
            render_route_card(route, i)
            with st.expander(f"  📍 View stop-by-stop details — Route {i}",
                             expanded=(i == 1)):
                render_leg_timeline(route)

    # ── Tab 2: Map ────────────────────────────────────────────────────────────
    with tab_map:
        st.markdown('<div class="section-title">Route Map</div>',
                    unsafe_allow_html=True)
        render_map(result["routes"])

        # Delay network map
        st.markdown('<div class="section-title">Station Delay Risk Network</div>',
                    unsafe_allow_html=True)
        try:
            from src.visualization.map_visualizer import RouteMapVisualizer
            import streamlit.components.v1 as components
            viz = RouteMapVisualizer()
            m   = viz.render_network()
            components.html(viz.get_html_string(m), height=380, scrolling=False)
        except Exception:
            st.info("Install `folium` for interactive maps.")

    # ── Tab 3: Fares ──────────────────────────────────────────────────────────
    with tab_fare:
        st.markdown('<div class="section-title">💰 Fare Comparison — Best Route</div>',
                    unsafe_allow_html=True)

        if best:
            dist  = best.get("total_distance_km", 500) or 500
            ttype = best.get("train_category", "Express")

            c1, c2, c3 = st.columns(3)
            with c1:
                sel_fare = best.get("fare_estimate", {})
                st.markdown(f"""
                <div class="stat-box">
                    <div class="value">₹{sel_fare.get('fare_per_passenger',0):,}</div>
                    <div class="label">{inputs['travel_class']} · Per Person</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="value">₹{sel_fare.get('total_fare',0):,}</div>
                    <div class="label">Total ({inputs['passengers']} pax)</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="value">₹{sel_fare.get('tatkal_total_fare',0):,}</div>
                    <div class="label">Tatkal Total</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("#### All Class Options")
            render_fare_table(result.get("fare_options", []), inputs["travel_class"])

            st.markdown(f"""
            <div class="alert-info" style="margin-top:10px">
                ℹ️ Fares based on official IR 2024 fare slabs · Distance: {dist} km ·
                Train type: {ttype} · Includes reservation charge + GST (AC classes)
            </div>""", unsafe_allow_html=True)

    # ── Tab 4: Availability ───────────────────────────────────────────────────
    with tab_avail:
        st.markdown('<div class="section-title">🎟 Seat Availability — Best Route</div>',
                    unsafe_allow_html=True)

        if result.get("availability"):
            render_availability(result["availability"], inputs["travel_class"])

        cal = result["context"]["calendar"] if result.get("context") else {}
        if cal:
            st.markdown(f"""
            <div class="context-banner" style="margin-top:12px">
                📊 Demand factor: <b>×{cal.get('occupancy_factor',1)}</b> &nbsp;|&nbsp;
                {cal.get('travel_advisory','—')}
            </div>""", unsafe_allow_html=True)

        # Best travel days
        st.markdown("#### 📆 Best Days to Travel (Next 7 Days)")
        try:
            from src.services.calendar_service import get_best_travel_days
            best_days = get_best_travel_days(datetime.date.today(), days_ahead=7)
            rows = [{
                "Date"       : d["date_str"],
                "Demand"     : d["risk_label"],
                "Occ. Factor": d["occupancy_factor"],
                "Delay Boost": f"+{d['delay_risk_boost']} min",
                "Note"       : d["note"],
            } for d in best_days]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load calendar data: {e}")

    # ── Tab 5: Explain AI ─────────────────────────────────────────────────────
    with tab_explain:
        st.markdown('<div class="section-title">🔍 AI Explainability</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-info" style="margin-bottom:12px">
            These charts explain <b>WHY</b> the model predicted the delay for the best route's
            destination. Green bars = features that reduce delay. Red bars = features that
            increase delay.
        </div>""", unsafe_allow_html=True)

        if best:
            render_explainability(best)

        # Feature importance bar
        st.markdown("#### 📊 Global Feature Importance")
        try:
            from src.explainability.shap_explainer import SHAPExplainer
            from src.utils.constants import MODEL_PKL
            exp = SHAPExplainer(MODEL_PKL)
            exp.build()
            fig = exp.plot_summary_bar()
            if fig:
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Feature importance chart unavailable: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()