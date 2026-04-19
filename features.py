"""
features.py
Clinically-motivated feature engineering for ICU vital sign time series.
Collapses per-patient hourly records into a single predictive feature vector.
"""

import numpy as np
import pandas as pd
from scipy import stats

VITAL_COLS = ["hr", "sbp", "dbp", "rr", "spo2", "temp", "gcs", "urine"]

# Clinical thresholds (NEWS2-based + ACCP/SCCM criteria)
THRESHOLDS = {
    "hr":    {"low": 50,  "high": 100},
    "sbp":   {"low": 90,  "high": 160},
    "dbp":   {"low": 60,  "high": 100},
    "rr":    {"low": 12,  "high": 20},
    "spo2":  {"low": 94,  "high": 100},
    "temp":  {"low": 36.0,"high": 38.5},
    "gcs":   {"low": 8,   "high": 15},
    "urine": {"low": 30,  "high": 200},
}

# NEWS2 scoring functions per vital
def news2_hr(hr):
    if hr <= 40 or hr >= 131: return 3
    if hr <= 50 or hr >= 111: return 1
    if hr >= 91:               return 1
    return 0

def news2_sbp(sbp):
    if sbp <= 90:              return 3
    if sbp <= 100:             return 2
    if sbp <= 110:             return 1
    if sbp >= 220:             return 3
    return 0

def news2_rr(rr):
    if rr <= 8 or rr >= 25:   return 3
    if rr <= 11:               return 1
    if rr >= 21:               return 2
    return 0

def news2_spo2(spo2):
    if spo2 <= 91:             return 3
    if spo2 <= 93:             return 2
    if spo2 <= 95:             return 1
    return 0

def news2_temp(temp):
    if temp <= 35.0:           return 3
    if temp <= 36.0:           return 1
    if temp >= 39.1:           return 2
    if temp >= 38.1:           return 1
    return 0

def news2_gcs(gcs):
    if gcs <= 8:               return 3
    if gcs <= 11:              return 2
    if gcs <= 14:              return 1
    return 0


def compute_trend_slope(series):
    """OLS slope of a vital over time — captures directional drift."""
    s = pd.Series(series).dropna()
    if len(s) < 3:
        return 0.0
    x = np.arange(len(s))
    slope, _, _, _, _ = stats.linregress(x, s)
    return round(slope, 4)


def compute_features_for_patient(patient_df):
    """
    Compute full feature vector for one patient from their hourly vital records.
    Uses last 24 hours of data (or all available if < 24h).
    """
    df = patient_df.sort_values("hour").tail(24).copy()

    # Impute missing values: forward fill then median
    for col in VITAL_COLS:
        df[col] = df[col].ffill().fillna(df[col].median())

    feats = {}

    # ── Static features ───────────────────────────────────────────────────
    feats["age"]       = df["age"].iloc[0]
    feats["gender"]    = df["gender"].iloc[0]
    feats["apache_ii"] = df["apache_ii"].iloc[0]
    feats["n_hours_observed"] = len(df)

    for v in VITAL_COLS:
        s = df[v].values

        # ── Summary stats ───────────────────────────────────────────────
        feats[f"{v}_mean"]   = np.nanmean(s)
        feats[f"{v}_std"]    = np.nanstd(s)
        feats[f"{v}_min"]    = np.nanmin(s)
        feats[f"{v}_max"]    = np.nanmax(s)
        feats[f"{v}_last"]   = s[-1]           # most recent reading
        feats[f"{v}_range"]  = np.nanmax(s) - np.nanmin(s)

        # ── Rolling windows (6h, 12h) ────────────────────────────────────
        for w in [6, 12]:
            window = s[-w:] if len(s) >= w else s
            feats[f"{v}_roll{w}_mean"] = np.nanmean(window)
            feats[f"{v}_roll{w}_std"]  = np.nanstd(window)

        # ── Trend slope ──────────────────────────────────────────────────
        feats[f"{v}_slope"] = compute_trend_slope(s)

        # ── Recent vs early difference (trajectory) ──────────────────────
        early = np.nanmean(s[:max(1, len(s)//3)])
        late  = np.nanmean(s[-max(1, len(s)//3):])
        feats[f"{v}_trajectory"] = late - early

        # ── Abnormality flags ─────────────────────────────────────────────
        lo = THRESHOLDS[v]["low"]
        hi = THRESHOLDS[v]["high"]
        feats[f"{v}_pct_abnormal"] = np.mean((s < lo) | (s > hi))
        feats[f"{v}_any_critical"]  = int(np.any((s < lo * 0.85) | (s > hi * 1.15)))

        # ── Time since last breach ────────────────────────────────────────
        breached = np.where((s < lo) | (s > hi))[0]
        feats[f"{v}_hours_since_breach"] = (len(s) - 1 - breached[-1]) if len(breached) else len(s)

    # ── Composite clinical scores ─────────────────────────────────────────
    last = {v: df[v].iloc[-1] for v in VITAL_COLS}

    # NEWS2 total
    feats["news2_total"] = (
        news2_hr(last["hr"]) +
        news2_sbp(last["sbp"]) +
        news2_rr(last["rr"]) +
        news2_spo2(last["spo2"]) +
        news2_temp(last["temp"]) +
        news2_gcs(last["gcs"])
    )

    # NEWS2 on rolling 6h means
    feats["news2_roll6"] = (
        news2_hr(feats["hr_roll6_mean"]) +
        news2_sbp(feats["sbp_roll6_mean"]) +
        news2_rr(feats["rr_roll6_mean"]) +
        news2_spo2(feats["spo2_roll6_mean"]) +
        news2_temp(feats["temp_roll6_mean"]) +
        news2_gcs(feats["gcs_roll6_mean"])
    )

    # Shock index (HR / SBP) — hemodynamic instability
    feats["shock_index"]      = last["hr"] / max(last["sbp"], 1)
    feats["shock_index_max"]  = feats["hr_max"] / max(feats["sbp_min"], 1)

    # Pulse pressure (SBP - DBP) — vascular tone
    feats["pulse_pressure"]   = last["sbp"] - last["dbp"]
    feats["pulse_pressure_min"] = feats["sbp_min"] - feats["dbp_max"]

    # Combined cardiorespiratory stress
    feats["hr_x_rr"]          = last["hr"] * last["rr"]
    feats["hr_div_spo2"]      = last["hr"] / max(last["spo2"], 1)

    # Hypoxic burden (sum of hours with SpO2 < 94)
    feats["hypoxic_hours"]    = int(np.sum(df["spo2"].values < 94))

    # NEWS2 trend (worsening?)
    feats["news2_slope"]      = feats["news2_total"] - feats["news2_roll6"]

    return feats


def build_feature_matrix(raw_df):
    """
    Apply feature engineering to all patients.
    Returns (X DataFrame, y Series).
    """
    print("Engineering features...")
    records = []
    patient_ids = []

    for pid, group in raw_df.groupby("patient_id"):
        feats = compute_features_for_patient(group)
        feats["outcome"] = group["outcome"].iloc[0]
        records.append(feats)
        patient_ids.append(pid)

    feat_df = pd.DataFrame(records, index=patient_ids)
    feat_df.index.name = "patient_id"

    X = feat_df.drop(columns=["outcome"])
    y = feat_df["outcome"]

    # Final NaN imputation (safety net)
    X = X.fillna(X.median())

    print(f"Feature matrix: {X.shape[0]} patients × {X.shape[1]} features")
    print(f"Positive rate (deterioration): {y.mean():.1%}")
    return X, y


if __name__ == "__main__":
    raw = pd.read_csv("data/icu_vitals_raw.csv")
    X, y = build_feature_matrix(raw)
    X.to_csv("data/features.csv")
    y.to_csv("data/labels.csv")
    print("Saved features.csv and labels.csv")
