"""
inference.py
Clinical inference layer: translates model probability scores into
tiered risk levels with actionable clinical recommendations.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH   = Path("models/best_model.pkl")
FEATURE_PATH = Path("models/feature_names.pkl")


# ── Clinical risk tier definitions ──────────────────────────────────────────
RISK_TIERS = {
    "Low": {
        "threshold": (0.0, 0.30),
        "color": "#3fb950",
        "action": "Routine monitoring. Reassess in 4 hours.",
        "escalation": None,
        "news2_alert": False,
    },
    "Medium": {
        "threshold": (0.30, 0.65),
        "color": "#ffa657",
        "action": "Increase monitoring frequency. Notify charge nurse. Reassess in 1 hour.",
        "escalation": "Charge nurse notification",
        "news2_alert": True,
    },
    "High": {
        "threshold": (0.65, 1.01),
        "color": "#f78166",
        "action": "IMMEDIATE physician alert. Consider Rapid Response Team activation.",
        "escalation": "Rapid Response Team",
        "news2_alert": True,
    },
}


def score_to_tier(score: float) -> str:
    if score < 0.30:   return "Low"
    elif score < 0.65: return "Medium"
    else:              return "High"


def generate_clinical_alert(patient_features: dict, risk_score: float,
                             top_features: list) -> dict:
    """
    Generate a structured clinical alert for a single patient.
    Returns a dict suitable for dashboard display or logging.
    """
    tier = score_to_tier(risk_score)
    tier_info = RISK_TIERS[tier]
    news2 = patient_features.get("news2_total", 0)

    # Build alert reasons from top contributing features
    reasons = []
    vital_labels = {
        "hr":   "Heart rate",   "sbp":  "Systolic BP",
        "rr":   "Respiratory rate", "spo2": "SpO2",
        "temp": "Temperature",  "gcs":  "GCS",
        "shock_index": "Shock index (HR/SBP)",
        "news2_total": "NEWS2 score",
        "apache_ii": "APACHE-II score",
    }
    for feat_name in top_features[:5]:
        val = patient_features.get(feat_name)
        if val is None:
            continue
        base = feat_name.split("_")[0]
        label = vital_labels.get(feat_name, vital_labels.get(base, feat_name))
        if "slope" in feat_name:
            direction = "rising" if val > 0 else "falling"
            reasons.append(f"{label} trending {direction} ({val:+.3f}/hr)")
        elif "std" in feat_name:
            reasons.append(f"{label} high variability (σ={val:.2f})")
        elif "pct_abnormal" in feat_name:
            reasons.append(f"{label} abnormal {val*100:.0f}% of observed time")
        elif "news2" in feat_name:
            reasons.append(f"NEWS2 score elevated: {int(val)}")
        elif "shock" in feat_name:
            reasons.append(f"Shock index: {val:.2f} (normal <0.7)")
        else:
            reasons.append(f"{label}: {val:.1f}")

    alert = {
        "risk_score":    round(risk_score, 4),
        "risk_tier":     tier,
        "risk_color":    tier_info["color"],
        "recommended_action": tier_info["action"],
        "escalation_target":  tier_info["escalation"],
        "news2_score":   int(news2),
        "top_risk_factors": reasons,
        "requires_immediate_action": tier == "High",
        "alert_message": _build_alert_message(tier, risk_score, news2, reasons),
    }
    return alert


def _build_alert_message(tier, score, news2, reasons):
    lines = [
        f"[{tier.upper()} RISK] Deterioration probability: {score*100:.1f}%",
        f"NEWS2 score: {int(news2)}",
        "",
        "Key risk factors:",
    ]
    for r in reasons[:4]:
        lines.append(f"  · {r}")
    lines.append("")
    lines.append(RISK_TIERS[tier]["action"])
    return "\n".join(lines)


class ICURiskPredictor:
    """Main inference class — load once, predict many."""

    def __init__(self, model_path=MODEL_PATH, feature_path=FEATURE_PATH):
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(feature_path)
        # Get feature importances for alert generation
        clf = self.model.named_steps["clf"]
        self.feature_importances = dict(
            zip(self.feature_names, clf.feature_importances_)
        )

    def predict_patient(self, feature_dict: dict) -> dict:
        """
        Predict risk for a single patient given their feature dictionary.
        Returns full clinical alert with risk tier and recommendations.
        """
        row = {f: feature_dict.get(f, np.nan) for f in self.feature_names}
        X = pd.DataFrame([row])[self.feature_names].fillna(0)

        risk_score = self.model.predict_proba(X)[0, 1]

        # Top features by importance that have notable values
        top_feats = sorted(self.feature_importances,
                           key=lambda f: self.feature_importances[f], reverse=True)[:10]

        return generate_clinical_alert(feature_dict, risk_score, top_feats)

    def predict_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict risk for a batch of patients (feature matrix)."""
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        scores = self.model.predict_proba(X_aligned)[:, 1]
        tiers  = [score_to_tier(s) for s in scores]
        return pd.DataFrame({
            "patient_id":  X.index,
            "risk_score":  np.round(scores, 4),
            "risk_tier":   tiers,
        })


if __name__ == "__main__":
    import json
    predictor = ICURiskPredictor()
    # Demo single patient
    sample = {
        "hr_last": 118, "hr_slope": 0.8, "hr_roll6_mean": 112,
        "rr_last": 24,  "rr_slope": 0.5,
        "spo2_last": 92, "spo2_trajectory": -3.2,
        "sbp_last": 94, "shock_index": 1.25,
        "gcs_last": 12, "news2_total": 9,
        "apache_ii": 22, "age": 71,
    }
    alert = predictor.predict_patient(sample)
    print(json.dumps({k: v for k, v in alert.items() if k != "risk_color"}, indent=2))
