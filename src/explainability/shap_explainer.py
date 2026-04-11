"""
=============================================================
  RouteMATE_AI — SHAP Explainer
  File: src/explainability/shap_explainer.py
=============================================================
  Explains WHY the model predicted a certain delay using SHAP
  (SHapley Additive exPlanations).

  SHAP answers: "Which features pushed the prediction
  UP or DOWN from the baseline?"

  Usage:
    from src.explainability.shap_explainer import SHAPExplainer

    explainer = SHAPExplainer()
    explainer.build(model_bundle)               # run once
    result = explainer.explain(feature_row)     # per prediction
    explainer.plot_waterfall(result)            # matplotlib chart
    text   = explainer.text_explanation(result) # plain English

  Requirements:
    pip install shap matplotlib
=============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Run: pip install shap")
    print("   Falling back to feature-importance based explanations.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

from src.utils.constants import (
    FEATURE_COLS, FEATURE_DISPLAY_NAMES, MODEL_PKL, SHAP_EXPLAINER_PKL
)


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP EXPLAINER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """
    Wraps a SHAP TreeExplainer around the Random Forest regressor.
    Falls back to permutation-importance explanation if SHAP not installed.
    """

    def __init__(self, model_path: str = MODEL_PKL):
        with open(model_path, "rb") as f:
            self.bundle = pickle.load(f)

        self.model        = self.bundle["rf_regressor"]
        self.feature_cols = self.bundle["feature_cols"]
        self.station_lkp  = self.bundle["station_delay_lookup"]
        self.explainer    = None
        self.background   = None

    # ── Build explainer ───────────────────────────────────────────────────────

    def build(self, background_df: pd.DataFrame = None):
        """
        Initialise the SHAP TreeExplainer.

        Args:
            background_df: Optional DataFrame of background samples
                           (used to compute expected value baseline).
                           If None, uses a small synthetic background.
        """
        if not SHAP_AVAILABLE:
            print("⚠️  SHAP not available — using fallback explainer.")
            return self

        if background_df is not None:
            self.background = background_df[self.feature_cols].fillna(0)
        else:
            # Synthetic background: mean values across feature space
            self.background = pd.DataFrame([{
                "train_cat_code"        : 0,
                "stop_position_ratio"   : 0.5,
                "distance_km"           : 400,
                "halt_min"              : 5,
                "time_of_day_min"       : 720,
                "time_block_code"       : 0,
                "is_overnight"          : 0,
                "journey_day"           : 1,
                "running_days_count"    : 7,
                "is_daily"              : 1,
                "total_stops"           : 15,
                "station_pct_right_time": 70,
                "station_pct_slight"    : 15,
                "station_pct_sig"       : 15,
                "distance_band_code"    : 1,
            }])

        self.explainer = shap.TreeExplainer(
            self.model,
            data            = self.background,
            feature_names   = self.feature_cols,
            feature_perturbation = "interventional",
        )
        print("✅ SHAP TreeExplainer initialised.")
        return self

    def save(self, path: str = SHAP_EXPLAINER_PKL):
        """Save the built explainer to disk."""
        if self.explainer is None:
            print("⚠️  Explainer not built yet. Call .build() first.")
            return
        with open(path, "wb") as f:
            pickle.dump(self.explainer, f)
        print(f"💾 SHAP explainer saved → {path}")

    def load(self, path: str = SHAP_EXPLAINER_PKL):
        """Load a pre-built explainer from disk."""
        with open(path, "rb") as f:
            self.explainer = pickle.load(f)
        return self

    # ── Explain a single prediction ───────────────────────────────────────────

    def explain(self, feature_row: dict) -> dict:
        """
        Compute SHAP values for a single prediction.

        Args:
            feature_row: dict of {feature_name: value} — same format
                         as output of helpers.build_feature_row()

        Returns dict with:
            predicted_value     : float
            base_value          : float (model average)
            shap_values         : dict {feature: shap_value}
            top_contributors    : list of (feature, value, shap) sorted by |shap|
            explanation_text    : plain English explanation
        """
        X = pd.DataFrame([feature_row])[self.feature_cols].fillna(0)
        predicted = float(self.model.predict(X)[0])

        if SHAP_AVAILABLE and self.explainer is not None:
            return self._explain_shap(X, predicted, feature_row)
        else:
            return self._explain_fallback(X, predicted, feature_row)

    def _explain_shap(self, X: pd.DataFrame, predicted: float, feature_row: dict) -> dict:
        """Full SHAP explanation."""
        sv = self.explainer(X)
        shap_vals   = sv.values[0]
        base_value  = float(sv.base_values[0])

        shap_dict = {
            col: round(float(sv), 3)
            for col, sv in zip(self.feature_cols, shap_vals)
        }

        # Sort by absolute SHAP value
        top = sorted(
            [(col, feature_row.get(col, 0), shap_dict[col]) for col in self.feature_cols],
            key=lambda x: abs(x[2]),
            reverse=True,
        )

        return {
            "predicted_value"  : round(predicted, 1),
            "base_value"       : round(base_value, 1),
            "shap_values"      : shap_dict,
            "top_contributors" : top[:8],
            "explanation_text" : self._build_explanation_text(predicted, base_value, top[:5]),
            "method"           : "SHAP TreeExplainer",
        }

    def _explain_fallback(self, X: pd.DataFrame, predicted: float, feature_row: dict) -> dict:
        """
        Fallback when SHAP is not installed:
        Uses Random Forest feature importances as proxy importance scores,
        weighted by deviation from typical values.
        """
        importances = self.model.feature_importances_

        typical = {
            "train_cat_code": 0, "stop_position_ratio": 0.5,
            "distance_km": 400, "halt_min": 5, "time_of_day_min": 720,
            "time_block_code": 0, "is_overnight": 0, "journey_day": 1,
            "running_days_count": 7, "is_daily": 1, "total_stops": 15,
            "station_pct_right_time": 70, "station_pct_slight": 15,
            "station_pct_sig": 15, "distance_band_code": 1,
        }

        pseudo_shap = {}
        for col, imp in zip(self.feature_cols, importances):
            val     = feature_row.get(col, 0) or 0
            typ     = typical.get(col, 0)
            scale   = max(abs(val - typ), 0.01)
            direction = 1 if val > typ else -1
            pseudo_shap[col] = round(imp * scale * direction * 10, 3)

        top = sorted(
            [(col, feature_row.get(col, 0), pseudo_shap[col]) for col in self.feature_cols],
            key=lambda x: abs(x[2]),
            reverse=True,
        )

        base = 141.0   # approximate dataset mean delay
        return {
            "predicted_value"  : round(predicted, 1),
            "base_value"       : round(base, 1),
            "shap_values"      : pseudo_shap,
            "top_contributors" : top[:8],
            "explanation_text" : self._build_explanation_text(predicted, base, top[:5]),
            "method"           : "Feature Importance Proxy (install shap for full SHAP)",
        }

    # ── Plain-English explanation ─────────────────────────────────────────────

    def _build_explanation_text(
        self,
        predicted  : float,
        base_value : float,
        top_contribs: list,
    ) -> str:
        """
        Generate a plain English explanation of the prediction.
        Example:
          "Predicted delay: 230 min. The biggest drivers are:
           Station Major Delay % (+85 min above baseline),
           Distance (km) (+42 min), Journey Progress % (-12 min)."
        """
        diff  = predicted - base_value
        lines = [f"Predicted delay: {predicted:.0f} min "
                 f"({'above' if diff >= 0 else 'below'} baseline of {base_value:.0f} min)."]

        lines.append("Key factors:")
        for col, val, sv in top_contribs:
            if abs(sv) < 0.5:
                continue
            fname = FEATURE_DISPLAY_NAMES.get(col, col)
            direction = "increased" if sv > 0 else "reduced"
            lines.append(
                f"  • {fname} = {round(val, 2)} → {direction} delay "
                f"by {abs(sv):.1f} min"
            )

        return "\n".join(lines)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_waterfall(
        self,
        explanation : dict,
        title       : str = "SHAP Waterfall — Delay Prediction",
        save_path   : str = None,
        max_features: int = 10,
    ):
        """
        Render a SHAP waterfall chart showing how each feature
        pushes the prediction above/below the baseline.

        Works with or without the shap library (uses matplotlib directly).
        """
        if not MPL_AVAILABLE:
            print("⚠️  matplotlib not available for plotting.")
            return None

        shap_vals = explanation["shap_values"]
        base      = explanation["base_value"]
        predicted = explanation["predicted_value"]

        # Sort features by absolute SHAP value
        sorted_items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        sorted_items = sorted_items[:max_features]
        sorted_items = sorted_items[::-1]   # bottom → top in chart

        labels = [FEATURE_DISPLAY_NAMES.get(k, k) for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

        fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.55)))

        # Running total bars
        running = base
        for i, (val, color, label) in enumerate(zip(values, colors, labels)):
            ax.barh(i, val, left=running, color=color, alpha=0.85, height=0.6)
            running += val
            ax.text(
                running + (1 if val >= 0 else -1),
                i,
                f"{val:+.1f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
                color=color,
            )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=base, color="grey", linestyle="--", linewidth=1, alpha=0.7, label=f"Baseline: {base:.0f} min")
        ax.axvline(x=predicted, color="#e67e22", linestyle="-", linewidth=2, label=f"Predicted: {predicted:.0f} min")

        red_patch   = mpatches.Patch(color="#e74c3c", label="Increases delay")
        green_patch = mpatches.Patch(color="#2ecc71", label="Reduces delay")
        ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=8)

        ax.set_xlabel("Delay (minutes)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 SHAP waterfall saved → {save_path}")

        return fig

    def plot_summary_bar(
        self,
        save_path: str = None,
    ):
        """
        Bar chart of global feature importance from the RF model.
        Works without SHAP.
        """
        if not MPL_AVAILABLE:
            return None

        importances = self.model.feature_importances_
        labels      = [FEATURE_DISPLAY_NAMES.get(c, c) for c in self.feature_cols]

        sorted_idx  = np.argsort(importances)
        sorted_imp  = importances[sorted_idx]
        sorted_lbl  = [labels[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(9, 7))
        colors  = plt.cm.RdYlGn_r(sorted_imp / sorted_imp.max())
        ax.barh(range(len(sorted_lbl)), sorted_imp, color=colors, alpha=0.85)
        ax.set_yticks(range(len(sorted_lbl)))
        ax.set_yticklabels(sorted_lbl, fontsize=9)
        ax.set_xlabel("Feature Importance", fontsize=10)
        ax.set_title("Global Feature Importance — Delay Prediction Model", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from src.utils.helpers import build_feature_row

    print("=" * 55)
    print("  SHAP Explainer — Test")
    print("=" * 55)

    explainer = SHAPExplainer()
    explainer.build()

    # Test: high-delay station (GHY)
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

    result_ghy = explainer.explain(row_ghy)
    print(f"\n  Station: GHY (Guwahati)")
    print(f"  Method: {result_ghy['method']}")
    print(f"  Predicted: {result_ghy['predicted_value']} min")
    print(f"  Baseline:  {result_ghy['base_value']} min")
    print(f"\n  Explanation:\n{result_ghy['explanation_text']}")

    # Test: low-delay station (KOAA)
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
    print(f"\n  Top contributors:")
    for feat, val, sv in result_koaa["top_contributors"][:5]:
        fname = FEATURE_DISPLAY_NAMES.get(feat, feat)
        print(f"    {fname:<35}: {sv:+.2f} min")

    # Save waterfall chart
    fig = explainer.plot_waterfall(result_ghy, title="SHAP — GHY Delay Explanation",
                                   save_path="/mnt/user-data/outputs/shap_waterfall_GHY.png")
    fig2 = explainer.plot_summary_bar(save_path="/mnt/user-data/outputs/shap_feature_importance.png")
 
    print("\n  ✅ SHAP charts saved.")
 