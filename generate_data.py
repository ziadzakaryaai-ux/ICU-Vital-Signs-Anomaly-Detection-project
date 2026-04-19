"""
generate_data.py
Generates realistic synthetic ICU vital sign dataset for 2,000 patients.
Distributions based on published clinical literature (PhysioNet 2012 stats).
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_PATIENTS = 2000
MAX_HOURS = 48
OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

# ── Clinical normal ranges ──────────────────────────────────────────────────
VITALS_CONFIG = {
    "hr":     {"normal": (75, 12),  "abnormal": (115, 20),  "low_thresh": 50,  "high_thresh": 100},
    "sbp":    {"normal": (120, 15), "abnormal": (88, 18),   "low_thresh": 90,  "high_thresh": 160},
    "dbp":    {"normal": (75, 10),  "abnormal": (55, 12),   "low_thresh": 60,  "high_thresh": 100},
    "rr":     {"normal": (16, 3),   "abnormal": (26, 5),    "low_thresh": 12,  "high_thresh": 20},
    "spo2":   {"normal": (97, 1.5), "abnormal": (91, 3),    "low_thresh": 94,  "high_thresh": 100},
    "temp":   {"normal": (37.0, 0.4),"abnormal": (38.8, 0.7),"low_thresh": 36.0,"high_thresh": 38.5},
    "gcs":    {"normal": (14.5, 0.8),"abnormal": (10, 3),   "low_thresh": 8,   "high_thresh": 15},
    "urine":  {"normal": (60, 20),  "abnormal": (25, 15),   "low_thresh": 30,  "high_thresh": 200},
}

def generate_vital_series(config, n_hours, deteriorating, deterioration_onset):
    """Generate a single vital sign time series with optional deterioration."""
    mu_n, sd_n = config["normal"]
    mu_a, sd_a = config["abnormal"]

    series = []
    for h in range(n_hours):
        if deteriorating and h >= deterioration_onset:
            # Gradual drift toward abnormal after onset
            progress = min(1.0, (h - deterioration_onset) / 12.0)
            mu = mu_n + progress * (mu_a - mu_n)
            sd = sd_n + progress * (sd_a - sd_n)
        else:
            mu, sd = mu_n, sd_n
        val = np.random.normal(mu, sd)
        # Small measurement noise
        val += np.random.normal(0, sd * 0.05)
        series.append(round(val, 1))
    return series


def build_dataset():
    rows = []
    patient_meta = []

    for pid in range(N_PATIENTS):
        # 30% of patients deteriorate (realistic ICU adverse event rate ~15-30%)
        deteriorating = np.random.random() < 0.28
        n_hours = np.random.randint(12, MAX_HOURS + 1)
        onset = np.random.randint(int(n_hours * 0.4), int(n_hours * 0.8)) if deteriorating else n_hours

        age = int(np.random.normal(63, 16))
        age = max(18, min(95, age))
        gender = np.random.choice(["M", "F"])
        apache_ii = int(np.random.normal(18, 7) if deteriorating else np.random.normal(12, 5))
        apache_ii = max(0, min(71, apache_ii))

        vitals_series = {v: generate_vital_series(cfg, n_hours, deteriorating, onset)
                         for v, cfg in VITALS_CONFIG.items()}

        for h in range(n_hours):
            # Introduce ~8% random missing values (realistic ICU documentation gaps)
            row = {
                "patient_id": f"P{pid:04d}",
                "hour": h,
                "age": age,
                "gender": 1 if gender == "M" else 0,
                "apache_ii": apache_ii,
                "outcome": int(deteriorating),
            }
            for v in VITALS_CONFIG:
                val = vitals_series[v][h]
                row[v] = np.nan if np.random.random() < 0.08 else val
            rows.append(row)

        patient_meta.append({
            "patient_id": f"P{pid:04d}",
            "age": age,
            "gender": gender,
            "apache_ii": apache_ii,
            "n_hours": n_hours,
            "deterioration_onset_hour": onset if deteriorating else None,
            "outcome": int(deteriorating),
        })

    df = pd.DataFrame(rows)
    meta = pd.DataFrame(patient_meta)
    df.to_csv(OUT_DIR / "icu_vitals_raw.csv", index=False)
    meta.to_csv(OUT_DIR / "patient_meta.csv", index=False)
    print(f"Generated {N_PATIENTS} patients, {len(df):,} hourly records.")
    print(f"Deterioration rate: {meta['outcome'].mean():.1%}")
    print(f"Saved to {OUT_DIR}/")
    return df, meta


if __name__ == "__main__":
    build_dataset()
