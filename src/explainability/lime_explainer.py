"""
=============================================================
  RouteMATE_AI — LIME Explainer
  File: src/explainability/lime_explainer.py
=============================================================
  Explains individual predictions using LIME
  (Local Interpretable Model-agnostic Explanations).

  LIME answers: "What is a simple LOCAL rule that approximates
  the model's behaviour around this specific prediction?"

  While SHAP is global (Shapley values), LIME is local:
  it perturbs the input and fits a simple linear model
  to understand the prediction neighbourhood.

  Usage:
    from src.explainability.lime_explainer import LIMEExplainer

    explainer = LIMEExplainer()
    explainer.build(training_df)
    result  = explainer.explain(feature_row)
    text    = result["explanation_text"]
    fig     = explainer.plot(result)

  Requirements:
    pip install lime matplotlib
=============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  LIME not installed. Run: pip install lime")
    print("   Falling back to sensitivity-based local explanation.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from src.utils.constants import (
    FEATURE_COLS, FEATURE_DISPLAY_NAMES, MODEL_PKL
)


# ══════════════════════════════════════════════════════════════════════════════
#  LIME EXPLAINER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class LIMEExplainer:
    """
    Wraps a LIME TabularExplainer around the Random Forest regressor.
    Falls back to local sensitivity analysis if LIME not installed.
    """

    def __init__(self, model_path: str = MODEL_PKL):
        with open(model_path, "rb") as f:
            self.bundle = pickle.load(f)

        self.model        = self.bundle["rf_regressor"]
        self.feature_cols = self.bundle["feature_cols"]
        self.station_lkp  = self.bundle["station_delay_lookup"]
        self.lime_exp     = None
        self.training_X   = None

        # Categorical feature indices (LIME needs to know these)
        self.categorical_features = [
            self.feature_cols.index(c) for c in [
                "train_cat_code", "time_block_code", "is_overnight",
                "is_daily", "distance_band_code", "journey_day",
            ] if c in self.feature_cols
        ]

        self.categorical_names = {
            self.feature_cols.index("train_cat_code")   : ["Express", "Passenger", "Superfast"],
            self.feature_cols.index("time_block_code")  : ["Morning", "Afternoon", "Evening", "Night", "Unknown"],
            self.feature_cols.index("is_overnight")     : ["No", "Yes"],
            self.feature_cols.index("is_daily")         : ["Not Daily", "Daily"],
            self.feature_cols.index("distance_band_code"): ["Short", "Medium", "Long"],
        } if LIME_AVAILABLE else {}

    # ── Build explainer ───────────────────────────────────────────────────────

    def build(self, training_df: pd.DataFrame = None):
        """
        Initialise LIME TabularExplainer with training data distribution.

        Args:
            training_df: DataFrame containing the training features.
                         If None, uses a synthetic representative background.
        """
        if not LIME_AVAILABLE:
            print("⚠️  LIME not available — using sensitivity fallback.")
            self._build_synthetic_background()
            return self

        if training_df is not None:
            self.training_X = training_df[self.feature_cols].fillna(0).values
        else:
            self.training_X = self._synthetic_background_array()

        self.lime_exp = lime_tabular.LimeTabularExplainer(
            training_data        = self.training_X,
            feature_names        = self.feature_cols,
            categorical_features = self.categorical_features,
            categorical_names    = self.categorical_names,
            mode                 = "regression",
            random_state         = 42,
        )
        print("✅ LIME TabularExplainer initialised.")
        return self

    def _build_synthetic_background(self):
        """Build a synthetic background array for fallback mode."""
        self.training_X = self._synthetic_background_array()

    def _synthetic_background_array(self) -> np.ndarray:
        """
        50 synthetic training samples representing the feature distribution.
        Used when real training data is not passed in.
        """
        np.random.seed(42)
        n = 200
        rows = []
        for _ in range(n):
            rows.append([
                np.random.choice([0, 1, 2]),                 # train_cat_code
                np.random.uniform(0, 1),                     # stop_position_ratio
                np.random.choice([50, 200, 500, 1000, 2000]),# distance_km
                np.random.choice([0, 2, 5, 10, 30]),         # halt_min
                np.random.uniform(0, 1439),                  # time_of_day_min
                np.random.choice([0, 1, 2, 3]),              # time_block_code
                np.random.choice([0, 1], p=[0.7, 0.3]),      # is_overnight
                np.random.choice([1, 2]),                    # journey_day
                np.random.randint(1, 8),                     # running_days_count
                np.random.choice([0, 1]),                    # is_daily
                np.random.randint(3, 30),                    # total_stops
                np.random.uniform(10, 95),                   # station_pct_right_time
                np.random.uniform(2, 40),                    # station_pct_slight
                np.random.uniform(2, 70),                    # station_pct_sig
                np.random.choice([0, 1, 2]),                 # distance_band_code
            ])
        return np.array(rows, dtype=float)

    # ── Explain a single prediction ───────────────────────────────────────────

    def explain(
        self,
        feature_row  : dict,
        num_features : int = 8,
        num_samples  : int = 1000,
    ) -> dict:
        """
        Generate a LIME explanation for a single prediction.

        Args:
            feature_row  : dict of feature values (from helpers.build_feature_row)
            num_features : top N features to show in explanation
            num_samples  : perturbation samples (more = more stable)

        Returns dict with:
            predicted_value     : float
            local_model_weights : {feature: weight}
            top_contributors    : list of (feature, value, weight)
            explanation_text    : plain English
            intercept           : local linear model intercept
        """
        X = np.array([[feature_row.get(c, 0) or 0 for c in self.feature_cols]])
        predicted = float(self.model.predict(X)[0])

        if LIME_AVAILABLE and self.lime_exp is not None:
            return self._explain_lime(X[0], predicted, feature_row, num_features, num_samples)
        else:
            return self._explain_sensitivity(X[0], predicted, feature_row, num_features)

    def _explain_lime(
        self,
        x_row       : np.ndarray,
        predicted   : float,
        feature_row : dict,
        num_features: int,
        num_samples : int,
    ) -> dict:
        """Full LIME explanation."""
        exp = self.lime_exp.explain_instance(
            data_row       = x_row,
            predict_fn     = self.model.predict,
            num_features   = num_features,
            num_samples    = num_samples,
        )

        # Extract local weights
        lime_weights = dict(exp.as_list())

        # Map back to clean feature names
        weights_clean = {}
        for rule_str, weight in lime_weights.items():
            # LIME returns rules like "distance_km > 400.00"
            # Extract the feature name from the rule
            matched = None
            for col in self.feature_cols:
                if col in rule_str:
                    matched = col
                    break
            if matched:
                weights_clean[matched] = round(weight, 3)

        top = sorted(
            [(col, feature_row.get(col, 0), weights_clean.get(col, 0))
             for col in self.feature_cols if col in weights_clean],
            key=lambda x: abs(x[2]),
            reverse=True,
        )

        intercept = round(exp.intercept[0] if hasattr(exp, "intercept") else predicted, 1)

        return {
            "predicted_value"   : round(predicted, 1),
            "intercept"         : intercept,
            "local_model_weights": weights_clean,
            "lime_raw"          : lime_weights,
            "top_contributors"  : top[:num_features],
            "explanation_text"  : self._build_text(predicted, top[:5]),
            "method"            : "LIME TabularExplainer",
        }

    def _explain_sensitivity(
        self,
        x_row       : np.ndarray,
        predicted   : float,
        feature_row : dict,
        num_features: int,
    ) -> dict:
        """
        Fallback: Local sensitivity analysis.
        Perturbs each feature ±10% and measures prediction change.
        This is a simple but effective proxy for LIME when unavailable.
        """
        sensitivities = {}

        for i, col in enumerate(self.feature_cols):
            val = x_row[i]
            if val == 0:
                delta = 1.0
            else:
                delta = abs(val) * 0.15

            # Perturb up
            x_up    = x_row.copy()
            x_up[i] = val + delta
            pred_up = float(self.model.predict(x_up.reshape(1, -1))[0])

            # Perturb down
            x_dn    = x_row.copy()
            x_dn[i] = val - delta
            pred_dn = float(self.model.predict(x_dn.reshape(1, -1))[0])

            # Local gradient: how much does prediction change per unit change
            sensitivity = (pred_up - pred_dn) / (2 * delta + 1e-9)
            sensitivities[col] = round(sensitivity, 3)

        top = sorted(
            [(col, feature_row.get(col, 0), sensitivities[col]) for col in self.feature_cols],
            key=lambda x: abs(x[2]),
            reverse=True,
        )

        return {
            "predicted_value"    : round(predicted, 1),
            "intercept"          : round(predicted, 1),
            "local_model_weights": sensitivities,
            "lime_raw"           : {},
            "top_contributors"   : top[:num_features],
            "explanation_text"   : self._build_text(predicted, top[:5]),
            "method"             : "Local Sensitivity Analysis (install lime for full LIME)",
        }

    # ── Plain-English explanation ─────────────────────────────────────────────

    def _build_text(self, predicted: float, top_contribs: list) -> str:
        """Generate a plain-English LIME explanation."""
        lines = [f"Local explanation for predicted delay: {predicted:.0f} min"]
        lines.append("Locally, these features matter most for this specific journey:")

        for col, val, weight in top_contribs:
            if abs(weight) < 0.01:
                continue
            fname     = FEATURE_DISPLAY_NAMES.get(col, col)
            direction = "raises" if weight > 0 else "lowers"
            lines.append(
                f"  • {fname} = {round(val, 2)} "
                f"→ locally {direction} delay (weight: {weight:+.2f})"
            )

        return "\n".join(lines)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(
        self,
        explanation : dict,
        title       : str  = "LIME Local Explanation",
        save_path   : str  = None,
        max_features: int  = 10,
    ):
        """
        Horizontal bar chart of LIME local weights.
        Red = increases delay, Green = reduces delay.
        """
        if not MPL_AVAILABLE:
            print("⚠️  matplotlib not available.")
            return None

        top = explanation["top_contributors"][:max_features][::-1]
        if not top:
            print("No contributors to plot.")
            return None

        labels  = [FEATURE_DISPLAY_NAMES.get(col, col) for col, _, _ in top]
        weights = [w for _, _, w in top]
        colors  = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]

        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.6)))
        bars = ax.barh(range(len(labels)), weights, color=colors, alpha=0.85, height=0.6)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Local Weight (effect on delay prediction)", fontsize=10)
        ax.set_title(
            f"{title}\nPredicted delay: {explanation['predicted_value']:.0f} min  |  Method: {explanation['method']}",
            fontsize=11, fontweight="bold"
        )

        # Value labels on bars
        for bar, w in zip(bars, weights):
            ax.text(
                w + (max(weights) * 0.02 if w >= 0 else -max(abs(w) for w in weights) * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{w:+.3f}",
                va="center",
                ha="left" if w >= 0 else "right",
                fontsize=8,
            )

        import matplotlib.patches as mpatches
        red_p   = mpatches.Patch(color="#e74c3c", label="Increases delay")
        green_p = mpatches.Patch(color="#2ecc71", label="Reduces delay")
        ax.legend(handles=[red_p, green_p], fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 LIME chart saved → {save_path}")

        return fig

    def compare_explanations(
        self,
        row_a    : dict,
        row_b    : dict,
        label_a  : str = "Route A",
        label_b  : str = "Route B",
        save_path: str = None,
    ):
        """
        Side-by-side LIME comparison of two routes.
        Useful for showing why Route A is better than Route B in the Streamlit app.
        """
        if not MPL_AVAILABLE:
            return None

        exp_a = self.explain(row_a)
        exp_b = self.explain(row_b)

        all_features = list({c for c, _, _ in exp_a["top_contributors"][:8]}
                           | {c for c, _, _ in exp_b["top_contributors"][:8]})

        wa = {c: w for c, _, w in exp_a["top_contributors"]}
        wb = {c: w for c, _, w in exp_b["top_contributors"]}

        w_a = [wa.get(f, 0) for f in all_features]
        w_b = [wb.get(f, 0) for f in all_features]
        labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in all_features]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, w_a, width, label=f"{label_a} ({exp_a['predicted_value']:.0f} min)", color="#3498db", alpha=0.8)
        ax.bar(x + width/2, w_b, width, label=f"{label_b} ({exp_b['predicted_value']:.0f} min)", color="#e67e22", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_ylabel("Local LIME Weight")
        ax.set_title("LIME Comparison: Two Routes", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 LIME comparison saved → {save_path}")

        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.utils.helpers import build_feature_row

    print("=" * 55)
    print("  LIME Explainer — Test")
    print("=" * 55)

    explainer = LIMEExplainer()
    explainer.build()   # uses synthetic background

    # High delay station
    row_ghy = build_feature_row(
        station_code    = "GHY",
        train_category  = "Express",
        stop_position   = 0.95,
        distance_km     = 1000,
        halt_min        = 10,
        time_of_day_min = 720,
        is_overnight    = 1,
        journey_day     = 2,
        running_days    = "MON",
        total_stops     = 20,
        station_pct_rt  = 10.0,
        station_pct_sl  = 20.0,
        station_pct_sig = 70.0,
    )

    result = explainer.explain(row_ghy, num_features=8)
    print(f"\n  Station: GHY (Guwahati)")
    print(f"  Method:    {result['method']}")
    print(f"  Predicted: {result['predicted_value']} min")
    print(f"\n  Explanation:\n{result['explanation_text']}")

    print("\n  Top contributors:")
    for col, val, w in result["top_contributors"][:6]:
        fname = FEATURE_DISPLAY_NAMES.get(col, col)
        print(f"    {fname:<35}: {w:+.3f}")

    # Low delay station
    row_koaa = build_feature_row(
        station_code    = "KOAA",
        train_category  = "Superfast",
        stop_position   = 0.0,
        distance_km     = 0,
        halt_min        = 0,
        time_of_day_min = 420,
        is_overnight    = 0,
        journey_day     = 1,
        running_days    = "MON,WED,SAT",
        total_stops     = 15,
        station_pct_rt  = 94.23,
        station_pct_sl  = 1.92,
        station_pct_sig = 3.85,
    )

    result_koaa = explainer.explain(row_koaa)
    print(f"\n  Station: KOAA (Kolkata)")
    print(f"  Predicted: {result_koaa['predicted_value']} min")

    # Save plots
    fig1 = explainer.plot(result, title="LIME — GHY Delay Explanation",
                          save_path="/mnt/user-data/outputs/lime_explanation_GHY.png")
    fig2 = explainer.compare_explanations(
        row_ghy, row_koaa,
        label_a="GHY (Express)", label_b="KOAA (Superfast)",
        save_path="/mnt/user-data/outputs/lime_comparison.png"
    )
    print("\n  ✅ LIME charts saved.")