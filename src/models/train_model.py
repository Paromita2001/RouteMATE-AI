"""
=============================================================
  RouteMATE_AI — Delay Prediction Model
  File: src/models/train_model.py
=============================================================
  What this does:
    1. FEATURE ENGINEERING  — build rich features from route data
    2. MODEL TRAINING       — Random Forest + XGBoost ensemble
    3. EVALUATION           — MAE, RMSE, R², cross-validation
    4. EXPLAINABILITY       — feature importance (SHAP-ready)
    5. SAVE MODEL           — model.pkl for predict.py to load
    6. ROUTE RISK SCORING   — score any route from route_engine

  Target variable:
    avg_delay_min  (continuous regression)
    delay_category (On Time / Slight / Significant — classification)

  Features used:
    - train_category       (Express / Passenger / Superfast)
    - stop_position_ratio  (how far into the journey)
    - distance_km          (cumulative distance at this stop)
    - halt_min             (scheduled stop duration)
    - time_of_day_min      (departure minute 0-1439)
    - time_block           (Morning/Afternoon/Evening/Night)
    - is_overnight         (crosses midnight)
    - journey_day          (day 1, 2, etc.)
    - running_days_count   (how many days/week it runs)
    - is_daily             (runs every day?)
    - total_stops          (total stops in train journey)
    - station_pct_right_time   (historical on-time % at station)
    - station_pct_slight_delay
    - station_pct_sig_delay
    - distance_band        (short / medium / long journey)
=============================================================
"""

import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble          import RandomForestClassifier
from sklearn.linear_model      import Ridge
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import cross_val_score, KFold
from sklearn.metrics           import (mean_absolute_error,
                                       mean_squared_error,
                                       r2_score,
                                       classification_report)
from sklearn.pipeline          import Pipeline

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "src",  "models")

# Dev fallback
if not os.path.exists(PROC_DIR):
    PROC_DIR  = "/mnt/user-data/outputs/preprocessed"
    MODEL_DIR = "/mnt/user-data/outputs"

os.makedirs(MODEL_DIR, exist_ok=True)

ROUTES_CSV = os.path.join(PROC_DIR, "train_routes_clean.csv")
DELAY_CSV  = os.path.join(PROC_DIR, "merged_delay.csv")
MODEL_PKL  = os.path.join(MODEL_DIR, "model.pkl")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(routes_df: pd.DataFrame, delay_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge route stops with station-level delay info and build ML features.
    Returns a clean feature matrix with target columns.
    """
    print("\n🔧 Engineering features...")

    df = routes_df.copy()

    # ── Merge station delay info ──────────────────────────────────────────────
    df = df.merge(
        delay_df[[
            "station_code", "avg_delay_min",
            "pct_right_time", "pct_slight_delay",
            "pct_significant_delay", "pct_cancelled_unknown",
            "delay_category",
        ]],
        on="station_code", how="left"
    )

    # ── Total stops per train ─────────────────────────────────────────────────
    train_stop_count = df.groupby("train_number")["stop_no"].max().rename("total_stops")
    df = df.merge(train_stop_count, on="train_number", how="left")

    # ── Stop position ratio (0 = start, 1 = end) ─────────────────────────────
    df["stop_position_ratio"] = df["stop_no"] / df["total_stops"].clip(lower=1)

    # ── Time of day block ─────────────────────────────────────────────────────
    def time_block(minutes):
        if pd.isna(minutes):
            return "Unknown"
        h = int(minutes) // 60
        if 5  <= h < 12: return "Morning"
        if 12 <= h < 17: return "Afternoon"
        if 17 <= h < 21: return "Evening"
        return "Night"

    df["time_block"]     = df["arrives_min"].apply(time_block)
    df["time_of_day_min"] = df["arrives_min"].fillna(df["departs_min"]).fillna(720)

    # ── Is overnight (multi-day journey) ─────────────────────────────────────
    df["is_overnight"] = (df["day"] > 1).astype(int)

    # ── Running days count ────────────────────────────────────────────────────
    def count_running_days(rdays):
        if pd.isna(rdays) or rdays == "":
            return 7
        return len(str(rdays).split(","))

    df["running_days_count"] = df["running_days"].apply(count_running_days)
    df["is_daily"]           = (df["running_days_count"] == 7).astype(int)

    # ── Halt duration (scheduled stop time at station) ────────────────────────
    df["halt_min"] = df["halt_min"].fillna(0).clip(upper=120)

    # ── Distance band ─────────────────────────────────────────────────────────
    def dist_band(km):
        if km <= 200:  return "Short"
        if km <= 800:  return "Medium"
        return "Long"

    df["distance_band"] = df["distance_km"].apply(dist_band)

    # ── Train category encode ─────────────────────────────────────────────────
    cat_map = {"Express": 0, "Passenger": 1, "Superfast": 2}
    df["train_cat_code"] = df["train_category"].map(cat_map).fillna(0)

    # ── Fill station delay NaN with overall median ────────────────────────────
    median_delay     = delay_df["avg_delay_min"].median()
    median_rt        = delay_df["pct_right_time"].median()
    median_slight    = delay_df["pct_slight_delay"].median()
    median_sig       = delay_df["pct_significant_delay"].median()

    df["station_avg_delay"]       = df["avg_delay_min"].fillna(median_delay)
    df["station_pct_right_time"]  = df["pct_right_time"].fillna(median_rt)
    df["station_pct_slight"]      = df["pct_slight_delay"].fillna(median_slight)
    df["station_pct_sig"]         = df["pct_significant_delay"].fillna(median_sig)

    # ── Encode categoricals for ML ────────────────────────────────────────────
    tb_map   = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3, "Unknown": 4}
    db_map   = {"Short": 0, "Medium": 1, "Long": 2}
    dc_map   = {"On Time": 0, "Slight Delay": 1, "Significantly Delayed": 2}

    df["time_block_code"]    = df["time_block"].map(tb_map).fillna(4)
    df["distance_band_code"] = df["distance_band"].map(db_map).fillna(1)
    df["delay_cat_code"]     = df["delay_category"].map(dc_map)   # target for classification

    print(f"   Total rows after feature engineering : {len(df):,}")
    print(f"   Rows with known delay (labelled)     : {df['station_avg_delay'].notna().sum():,}")

    return df


# ── Final feature columns ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "train_cat_code",
    "stop_position_ratio",
    "distance_km",
    "halt_min",
    "time_of_day_min",
    "time_block_code",
    "is_overnight",
    "journey_day",
    "running_days_count",
    "is_daily",
    "total_stops",
    "station_pct_right_time",
    "station_pct_slight",
    "station_pct_sig",
    "distance_band_code",
]

# Rename day → journey_day to avoid confusion
def add_journey_day(df):
    df["journey_day"] = df["day"].fillna(1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_models(df: pd.DataFrame):
    """
    Train regression + classification models on the feature matrix.
    Returns trained model objects and metadata.
    """
    print("\n🤖 Training models...")

    df = add_journey_day(df)

    # ── Filter to rows that have real delay labels ────────────────────────────
    labelled = df[df["avg_delay_min"].notna()].copy()
    print(f"   Labelled training samples: {len(labelled)}")

    if len(labelled) < 10:
        print("   ⚠️  Very few labelled samples — using augmentation strategy")
        labelled = _augment_training_data(df, labelled)
        print(f"   Augmented to: {len(labelled)} samples")

    X = labelled[FEATURE_COLS].fillna(0)
    y_reg  = labelled["station_avg_delay"]       # regression target
    y_clf  = labelled["delay_cat_code"].fillna(0).astype(int)  # classification target

    # ── Regression: Random Forest ─────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y_reg)

    # ── Regression: Gradient Boosting ─────────────────────────────────────────
    gb = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    gb.fit(X, y_reg)

    # ── Classification: Random Forest ────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y_clf)

    print("   ✅ RandomForest Regressor trained")
    print("   ✅ GradientBoosting Regressor trained")
    print("   ✅ RandomForest Classifier trained")

    return rf, gb, clf, X, y_reg, y_clf, labelled


def _augment_training_data(df: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """
    Since we have only 21 labelled stations, augment by:
    - sampling unlabelled stops and assigning synthetic delays
      based on distance + train category heuristics.
    This gives the model more variety to learn from.
    """
    np.random.seed(42)
    unlabelled = df[df["avg_delay_min"].isna()].sample(
        n=min(500, len(df[df["avg_delay_min"].isna()])),
        random_state=42
    ).copy()

    # Heuristic: delay grows with distance and is higher for Passenger trains
    base = unlabelled["distance_km"] * 0.05
    cat_bonus = unlabelled["train_cat_code"].map({0: 10, 1: 30, 2: 5}).fillna(10)
    noise = np.random.normal(0, 15, len(unlabelled))

    unlabelled["station_avg_delay"] = (base + cat_bonus + noise).clip(lower=0)
    unlabelled["avg_delay_min"]     = unlabelled["station_avg_delay"]

    def assign_cat(d):
        if d <= 15: return 0
        if d <= 60: return 1
        return 2
    unlabelled["delay_cat_code"] = unlabelled["avg_delay_min"].apply(assign_cat)

    return pd.concat([labelled, unlabelled], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_models(rf, gb, clf, X, y_reg, y_clf):
    print("\n📊 Evaluating models...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in [("RandomForest", rf), ("GradientBoosting", gb)]:
        preds = model.predict(X)
        mae   = mean_absolute_error(y_reg, preds)
        rmse  = np.sqrt(mean_squared_error(y_reg, preds))
        r2    = r2_score(y_reg, preds)
        cv    = cross_val_score(model, X, y_reg, cv=kf, scoring="neg_mean_absolute_error")
        print(f"\n   {name} Regressor:")
        print(f"     MAE  : {mae:.2f} min")
        print(f"     RMSE : {rmse:.2f} min")
        print(f"     R²   : {r2:.4f}")
        print(f"     CV MAE (5-fold): {-cv.mean():.2f} ± {cv.std():.2f}")

    print(f"\n   RandomForest Classifier:")
    clf_preds = clf.predict(X)
    label_names = ["On Time", "Slight Delay", "Significant Delay"]
    #print(classification_report(y_clf, clf_preds, target_names=label_names, zero_division=0))
    print(classification_report(y_clf, clf_preds, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_importance(rf, feature_cols):
    print("\n🔍 Feature Importance (Random Forest):")
    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    for feat, imp in importance.items():
        bar = "█" * int(imp * 40)
        print(f"   {feat:<30} {bar} {imp:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(rf, gb, clf, feature_cols, delay_df, path=MODEL_PKL):
    """Save all model artifacts to a single pickle file."""

    # Station delay lookup dict for fast inference
    # station_delay_lookup = delay_df.set_index("station_code")[
    #     ["avg_delay_min", "pct_right_time", "pct_slight_delay",
    #      "pct_significant_delay", "delay_category"]
    # ].to_dict(orient="index")
    
    delay_unique = delay_df.drop_duplicates(subset="station_code")

    station_delay_lookup = delay_unique.set_index("station_code")[  
        ["avg_delay_min", "pct_right_time", "pct_slight_delay",
         "pct_significant_delay", "delay_category"]
    ].to_dict(orient="index")

    bundle = {
        "rf_regressor"         : rf,
        "gb_regressor"         : gb,
        "rf_classifier"        : clf,
        "feature_cols"         : feature_cols,
        "station_delay_lookup" : station_delay_lookup,
        "label_map"            : {0: "On Time", 1: "Slight Delay", 2: "Significantly Delayed"},
        "cat_map"              : {"Express": 0, "Passenger": 1, "Superfast": 2},
        "time_block_map"       : {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3, "Unknown": 4},
        "distance_band_map"    : {"Short": 0, "Medium": 1, "Long": 2},
    }

    with open(path, "wb") as f:
        pickle.dump(bundle, f)

    size_kb = os.path.getsize(path) / 1024
    print(f"\n💾 Model saved → {path}  ({size_kb:.1f} KB)")
    return bundle


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTOR CLASS  (used by predict.py and app.py)
# ══════════════════════════════════════════════════════════════════════════════

class DelayPredictor:
    """
    Load saved model and predict delay for any station/train combination.
    Also scores full routes from route_engine output.
    """

    RISK_THRESHOLDS = {"low": 20, "medium": 60}  # minutes

    def __init__(self, model_path: str = MODEL_PKL):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.rf          = bundle["rf_regressor"]
        self.gb          = bundle["gb_regressor"]
        self.clf         = bundle["rf_classifier"]
        self.features    = bundle["feature_cols"]
        self.station_lkp = bundle["station_delay_lookup"]
        self.label_map   = bundle["label_map"]
        self.cat_map     = bundle["cat_map"]
        self.tb_map      = bundle["time_block_map"]
        self.db_map      = bundle["distance_band_map"]

    # ── Predict single stop ───────────────────────────────────────────────────

    def predict_stop(
        self,
        station_code    : str,
        train_category  : str  = "Express",
        stop_position   : float = 0.5,
        distance_km     : float = 500.0,
        halt_min        : float = 5.0,
        time_of_day_min : float = 720.0,
        is_overnight    : int   = 0,
        journey_day     : int   = 1,
        running_days    : int   = 7,
        total_stops     : int   = 15,
    ) -> dict:
        """Predict delay for a single train stop."""

        # Station historical features
        s = self.station_lkp.get(station_code.upper(), {})
        pct_rt     = s.get("pct_right_time",         50.0)
        pct_slight = s.get("pct_slight_delay",        25.0)
        pct_sig    = s.get("pct_significant_delay",   25.0)

        # Time block
        h = int(time_of_day_min) // 60
        if   5  <= h < 12: tb = "Morning"
        elif 12 <= h < 17: tb = "Afternoon"
        elif 17 <= h < 21: tb = "Evening"
        else:              tb = "Night"

        # Distance band
        if distance_km <= 200:   db = "Short"
        elif distance_km <= 800: db = "Medium"
        else:                    db = "Long"

        row = {
            "train_cat_code"        : self.cat_map.get(train_category, 0),
            "stop_position_ratio"   : stop_position,
            "distance_km"           : distance_km,
            "halt_min"              : halt_min,
            "time_of_day_min"       : time_of_day_min,
            "time_block_code"       : self.tb_map.get(tb, 4),
            "is_overnight"          : is_overnight,
            "journey_day"           : journey_day,
            "running_days_count"    : running_days,
            "is_daily"              : int(running_days == 7),
            "total_stops"           : total_stops,
            "station_pct_right_time": pct_rt,
            "station_pct_slight"    : pct_slight,
            "station_pct_sig"       : pct_sig,
            "distance_band_code"    : self.db_map.get(db, 1),
        }

        X = pd.DataFrame([row])[self.features]

        # Ensemble: average RF + GB predictions
        rf_pred  = self.rf.predict(X)[0]
        gb_pred  = self.gb.predict(X)[0]
        pred_min = round((rf_pred + gb_pred) / 2, 1)
        pred_min = max(0, pred_min)

        # Classification
        clf_code  = self.clf.predict(X)[0]
        clf_label = self.label_map.get(clf_code, "Unknown")

        # Risk label
        if pred_min <= self.RISK_THRESHOLDS["low"]:
            risk = "🟢 Low"
        elif pred_min <= self.RISK_THRESHOLDS["medium"]:
            risk = "🟡 Medium"
        else:
            risk = "🔴 High"

        return {
            "station_code"        : station_code.upper(),
            "predicted_delay_min" : pred_min,
            "predicted_delay_hrs" : round(pred_min / 60, 2),
            "delay_category"      : clf_label,
            "risk_label"          : risk,
            "historical_delay"    : s.get("avg_delay_min", None),
            "historical_category" : s.get("delay_category", "Unknown"),
        }


    # ── Score a full route ────────────────────────────────────────────────────

    def score_route(self, route: dict) -> dict:
        """
        Takes a single route dict from route_engine.smart_route_search()
        and returns it enriched with delay predictions + overall risk score.

        Usage:
            result = graph.smart_route_search("KOAA", "GHY")
            for route in result["routes"]:
                scored = predictor.score_route(route)
        """
        legs = route.get("legs", [])
        if not legs:
            return route

        leg_predictions = []
        total_predicted_delay = 0.0
        max_risk_level = 0   # 0=Low, 1=Medium, 2=High

        for leg in legs:
            station   = leg.get("to", "")
            cat       = leg.get("train_category", "Express")
            dist      = leg.get("distance_km",  500.0)
            travel    = leg.get("travel_min",   120.0) or 120.0
            dept_min  = leg.get("depart_time",  None)

            # Parse depart time to minutes
            tod = 720.0
            if dept_min and str(dept_min) not in ("None", "nan", "Source"):
                try:
                    h, m = str(dept_min).split(":")
                    tod = int(h) * 60 + int(m)
                except Exception:
                    pass

            pred = self.predict_stop(
                station_code    = station,
                train_category  = cat,
                distance_km     = dist,
                time_of_day_min = tod,
            )
            leg_predictions.append({**leg, "delay_prediction": pred})
            total_predicted_delay += pred["predicted_delay_min"]

            risk_order = {"🟢 Low": 0, "🟡 Medium": 1, "🔴 High": 2}
            level = risk_order.get(pred["risk_label"], 0)
            if level > max_risk_level:
                max_risk_level = level

        risk_labels = ["🟢 Low", "🟡 Medium", "🔴 High"]
        avg_delay   = round(total_predicted_delay / max(len(legs), 1), 1)

        # Overall route score (0-100, lower = better)
        base_score = min(avg_delay / 3, 100)
        change_penalty = route.get("changes", 0) * 10
        route_score = round(100 - min(base_score + change_penalty, 99), 1)

        return {
            **route,
            "legs"                    : leg_predictions,
            "predicted_avg_delay_min" : avg_delay,
            "overall_risk"            : risk_labels[max_risk_level],
            "route_score"             : route_score,   # 0-100, higher = smarter
            "score_label"             : _score_label(route_score),
        }


    def score_all_routes(self, search_result: dict) -> dict:
        """
        Score all routes in a smart_route_search result.
        Returns the result with routes sorted by route_score descending.
        """
        scored_routes = [self.score_route(r) for r in search_result.get("routes", [])]
        scored_routes.sort(key=lambda r: r.get("route_score", 0), reverse=True)

        return {
            **search_result,
            "routes": scored_routes,
        }


def _score_label(score: float) -> str:
    if score >= 80: return "⭐⭐⭐ Excellent"
    if score >= 60: return "⭐⭐ Good"
    if score >= 40: return "⭐ Fair"
    return "⚠️ High Risk"


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — train, evaluate, save
# ══════════════════════════════════════════════════════════════════════════════

def run_training():
    print("=" * 60)
    print("  RouteMATE_AI — Model Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n📂 Loading processed data...")
    routes = pd.read_csv(ROUTES_CSV)
    delay  = pd.read_csv(DELAY_CSV)
    print(f"   Routes: {routes.shape}  |  Delay stations: {len(delay)}")

    # Feature engineering
    df = engineer_features(routes, delay)
    df = add_journey_day(df)

    # Train
    rf, gb, clf, X, y_reg, y_clf, labelled = train_models(df)

    # Evaluate
    evaluate_models(rf, gb, clf, X, y_reg, y_clf)

    # Feature importance
    print_feature_importance(rf, FEATURE_COLS)

    # Save
    bundle = save_model(rf, gb, clf, FEATURE_COLS, delay)

    print("\n✅ Training complete!")
    print(f"   Model saved to: {MODEL_PKL}")
    return bundle


if __name__ == "__main__":
    bundle = run_training()

    # ── Quick inference test ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Quick Inference Test")
    print("=" * 60)

    predictor = DelayPredictor(MODEL_PKL)

    test_cases = [
        {"station_code": "KOAA",  "train_category": "Superfast", "distance_km": 0},
        {"station_code": "GHY",   "train_category": "Express",   "distance_km": 1000},
        {"station_code": "NJP",   "train_category": "Passenger", "distance_km": 600},
        {"station_code": "AGTL",  "train_category": "Express",   "distance_km": 1600},
        {"station_code": "MLDT",  "train_category": "Passenger", "distance_km": 350},
    ]

    print(f"\n{'Station':<8} {'Category':<12} {'Dist':>6}  {'Pred Delay':>12}  {'Risk':<14}  {'Category'}")
    print("-" * 75)
    for tc in test_cases:
        result = predictor.predict_stop(**tc)
        print(
            f"{result['station_code']:<8} "
            f"{tc['train_category']:<12} "
            f"{tc['distance_km']:>6} km "
            f"{result['predicted_delay_min']:>8.1f} min  "
            f"{result['risk_label']:<14}  "
            f"{result['delay_category']}"
        )