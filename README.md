[README.md](https://github.com/user-attachments/files/26868197/README.md)
# ICU Patient Deterioration Risk Prediction System
### Early Warning Score Model Using Vital Sign Dynamics & Clinical Features

> **A complete end-to-end clinical AI pipeline** — from raw ICU vital signs to tiered risk alerts with physician-facing explanations.

---

## Overview

This project implements a machine learning system that continuously monitors ICU patient vital signs and predicts the risk of clinical deterioration (e.g., hemodynamic instability, respiratory failure, or imminent need for escalation of care). It mirrors the architecture of real-world systems like the Epic Deterioration Index and MEWS-based early warning tools deployed in hospital settings.

**Dataset:** Synthetic ICU vitals generated from distributions matching the PhysioNet 2012 Clinical Challenge dataset (2,000 patients, ~60,000 hourly observations). To use real data, download from `physionet.org/content/challenge-2012` and point `generate_data.py` to it.

---

## Results

| Model | AUROC | AUPRC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Random Forest | **0.999** | 0.998 | 0.983 | 0.990 | 0.987 |
| Gradient Boosting | 0.998 | 0.998 | 0.974 | 0.998 | 0.986 |

*5-fold stratified cross-validation · Classification threshold = 0.32 (tuned for high recall)*

---

## Project Structure

```
icu_risk_project/
├── generate_data.py    # Synthetic ICU vital signs dataset (2,000 patients)
├── features.py         # Clinical feature engineering (134 features)
├── train.py            # Model training, evaluation & plots
├── inference.py        # Risk tier classification & clinical alert generation
├── dashboard.py        # Streamlit monitoring dashboard
├── data/
│   ├── icu_vitals_raw.csv      # Raw hourly vital signs
│   ├── features.csv            # Engineered feature matrix
│   └── patient_meta.csv        # Patient demographics & outcomes
├── models/
│   ├── best_model.pkl          # Trained model pipeline
│   ├── feature_names.pkl       # Feature registry
│   └── metrics.json            # Cross-validated performance metrics
└── outputs/
    ├── predictions.csv         # Patient-level risk scores & tiers
    ├── evaluation_report.png   # 6-panel evaluation visualization
    └── feature_importances.csv # Ranked feature importances
```

---

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap streamlit xgboost joblib scipy
```

---

## Usage

### Step 1 — Generate data & train model
```bash
python train.py
```
This runs the full pipeline: data generation → feature engineering → model training → evaluation plots → prediction export.

### Step 2 — Launch dashboard
```bash
streamlit run dashboard.py
```
Opens an interactive clinical monitoring dashboard with:
- Per-patient risk score + tier (Low / Medium / High)
- 6-panel vital sign trend charts with clinical normal ranges
- Top contributing features per patient
- Population-level risk distribution
- Full model evaluation report

### Step 3 — Run inference on a new patient
```python
from inference import ICURiskPredictor

predictor = ICURiskPredictor()
alert = predictor.predict_patient({
    "hr_last": 118, "hr_slope": 0.8,
    "rr_last": 24,  "spo2_last": 92,
    "sbp_last": 94, "shock_index": 1.25,
    "gcs_last": 12, "news2_total": 9,
    "apache_ii": 22, "age": 71,
})
print(alert["alert_message"])
```

---

## Feature Engineering

134 features engineered from 8 raw vital signs:

| Category | Features | Count |
|---|---|---|
| Summary statistics | Mean, std, min, max, last, range per vital | 48 |
| Rolling windows (6h, 12h) | Rolling mean & std | 32 |
| Trend slopes | Linear regression slope per vital | 8 |
| Trajectory | Early vs late mean difference | 8 |
| Abnormality flags | % time abnormal, any critical breach | 16 |
| Time features | Hours since last threshold breach | 8 |
| Composite scores | NEWS2, Shock Index, Pulse Pressure, HR×RR | 8 |
| Static | Age, Gender, APACHE-II, hours observed | 4 |

Key engineered features:
- **Shock Index** (HR/SBP) — validated hemodynamic instability marker
- **NEWS2 Score** — national early warning score, directly maps to real triage
- **Rolling variability** — high std deviation = physiological stress
- **Trend slope** — rising RR or falling SpO2 are ICU red flags

---

## Clinical Interpretation Layer

| Tier | Score | Action |
|---|---|---|
| 🟢 **Low** | < 0.30 | Routine monitoring. Reassess in 4h. |
| 🟡 **Medium** | 0.30–0.65 | Increase monitoring. Notify charge nurse. Reassess in 1h. |
| 🔴 **High** | > 0.65 | IMMEDIATE physician alert. Consider Rapid Response Team. |

Classification threshold is set at **0.32** (not the default 0.5) because in ICU settings, the clinical cost of a missed deterioration (false negative) far exceeds the cost of an unnecessary review (false positive).

---

## Why These Design Choices?

**Random Forest over neural networks:** With ~2,000 tabular patients, tree ensembles consistently match or beat neural networks. More importantly, feature importances enable SHAP-based explainability — a regulatory and trust requirement for clinical AI.

**Recall-first evaluation:** ICU adverse events are rare (~28% positive rate). Optimizing for accuracy would allow the model to predict "stable" for everyone and be 72% accurate. Recall ensures we catch the patients who matter.

**NEWS2 integration:** Incorporating the validated National Early Warning Score as a feature grounds the ML model in established clinical reasoning, making it defensible to physicians and ethics committees.

---

## Disclaimer

This project uses **synthetic data** and is intended for **educational and portfolio purposes only**. It is not a medical device and must not be used for clinical decision-making.
