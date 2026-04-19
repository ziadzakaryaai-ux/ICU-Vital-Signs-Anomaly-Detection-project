"""
train.py
Trains and evaluates ICU deterioration risk models.
Produces evaluation report, feature importance plot, and saved model artifacts.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

from generate_data import build_dataset
from features import build_feature_matrix

MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.32   # Tuned for high recall (clinical priority)


def train_and_evaluate():
    # ── 1. Data ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ICU PATIENT DETERIORATION RISK — TRAINING PIPELINE")
    print("=" * 60)
    raw_df, meta = build_dataset()
    X, y = build_feature_matrix(raw_df)

    feature_names = list(X.columns)

    # ── 2. Models ──────────────────────────────────────────────────────────
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                subsample=0.8, min_samples_leaf=10, random_state=42
            ))
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n── Cross-validated evaluation (5-fold stratified) ──\n")
    for name, pipeline in models.items():
        proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = (proba >= THRESHOLD).astype(int)

        auroc  = roc_auc_score(y, proba)
        auprc  = average_precision_score(y, proba)
        recall = recall_score(y, preds)
        prec   = precision_score(y, preds, zero_division=0)
        f1     = f1_score(y, preds)

        results[name] = {
            "proba": proba, "preds": preds,
            "auroc": auroc, "auprc": auprc,
            "recall": recall, "precision": prec, "f1": f1
        }

        print(f"{name}:")
        print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
              f"Recall={recall:.3f}  Precision={prec:.3f}  F1={f1:.3f}")
        print(f"  {classification_report(y, preds, target_names=['Stable','Deteriorating'], digits=3)}")

    # ── 3. Fit best model on full data ─────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auroc"])
    best_pipeline = models[best_name]
    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, MODELS_DIR / "best_model.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    # Save threshold & metrics
    metrics_out = {k: {m: round(v, 4) for m, v in v2.items() if m not in ("proba","preds")}
                   for k, v2 in results.items()}
    metrics_out["best_model"] = best_name
    metrics_out["threshold"] = THRESHOLD
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nBest model: {best_name} — saved.")

    # ── 4. Feature importances (top 20) ────────────────────────────────────
    clf = best_pipeline.named_steps["clf"]
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
    feat_imp.to_csv(OUTPUTS_DIR / "feature_importances.csv")

    # ── 5. Plots ────────────────────────────────────────────────────────────
    _make_evaluation_plots(results, feat_imp, y, best_name)

    # ── 6. Patient-level predictions CSV ───────────────────────────────────
    proba_best = results[best_name]["proba"]
    pred_df = pd.DataFrame({
        "patient_id": X.index,
        "risk_score": np.round(proba_best, 4),
        "risk_tier": pd.cut(proba_best, bins=[0, 0.30, 0.65, 1.0],
                            labels=["Low", "Medium", "High"]),
        "predicted_deterioration": (proba_best >= THRESHOLD).astype(int),
        "actual_outcome": y.values,
        "age": X["age"].values,
        "apache_ii": X["apache_ii"].values,
        "news2_total": X["news2_total"].values,
        "shock_index": np.round(X["shock_index"].values, 3),
    })
    pred_df.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)
    print(f"\nPredictions saved: {len(pred_df)} patients")
    print(pred_df["risk_tier"].value_counts().to_string())

    return results, feat_imp


def _make_evaluation_plots(results, feat_imp, y, best_name):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ACCENT  = "#58a6ff"
    ACCENT2 = "#3fb950"
    WARN    = "#f78166"
    MUTED   = "#8b949e"
    BG      = "#0d1117"
    CARD    = "#161b22"
    TEXT    = "#e6edf3"

    def style_ax(ax):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED)
        ax.title.set_color(TEXT)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        for spine in ax.spines.values(): spine.set_edgecolor('#30363d')

    colors_map = {"Random Forest": ACCENT, "Gradient Boosting": ACCENT2}

    # ── ROC curves ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    ax1.plot([0,1],[0,1],"--", color=MUTED, lw=1, alpha=0.5)
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y, res["proba"])
        ax1.plot(fpr, tpr, color=colors_map[name], lw=2,
                 label=f'{name} (AUC={res["auroc"]:.3f})')
    ax1.set_xlabel("False Positive Rate", color=MUTED, fontsize=10)
    ax1.set_ylabel("True Positive Rate", color=MUTED, fontsize=10)
    ax1.set_title("ROC Curve", color=TEXT, fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, edgecolor="#30363d")

    # ── Precision-Recall curves ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    baseline = y.mean()
    ax2.axhline(baseline, color=MUTED, lw=1, ls="--", alpha=0.5, label=f"Baseline ({baseline:.2f})")
    for name, res in results.items():
        prec_arr, rec_arr, _ = precision_recall_curve(y, res["proba"])
        ax2.plot(rec_arr, prec_arr, color=colors_map[name], lw=2,
                 label=f'{name} (AP={res["auprc"]:.3f})')
    ax2.set_xlabel("Recall", color=MUTED, fontsize=10)
    ax2.set_ylabel("Precision", color=MUTED, fontsize=10)
    ax2.set_title("Precision–Recall Curve", color=TEXT, fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, edgecolor="#30363d")

    # ── Feature importance ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3)
    top15 = feat_imp.head(15)
    bar_colors = [ACCENT2 if "news2" in n or "shock" in n or "apache" in n else ACCENT
                  for n in top15.index]
    ax3.barh(range(len(top15)), top15.values, color=bar_colors, alpha=0.85)
    ax3.set_yticks(range(len(top15)))
    ax3.set_yticklabels(top15.index, fontsize=9, color=TEXT)
    ax3.invert_yaxis()
    ax3.set_xlabel("Feature Importance", color=MUTED, fontsize=10)
    ax3.set_title(f"Top 15 Features\n({best_name})", color=TEXT, fontsize=12, fontweight="bold")

    # ── Confusion matrix (best model) ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4)
    res = results[best_name]
    cm = confusion_matrix(y, res["preds"])
    im = ax4.imshow(cm, cmap="Blues", aspect="auto")
    ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
    ax4.set_xticklabels(["Stable","Deteriorating"], color=TEXT, fontsize=10)
    ax4.set_yticklabels(["Stable","Deteriorating"], color=TEXT, fontsize=10)
    ax4.set_xlabel("Predicted", color=MUTED, fontsize=10)
    ax4.set_ylabel("Actual", color=MUTED, fontsize=10)
    ax4.set_title(f"Confusion Matrix\n({best_name}, threshold={results[best_name].get('threshold',THRESHOLD):.2f})",
                  color=TEXT, fontsize=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color=TEXT, fontsize=16, fontweight="bold")

    # ── Risk score distribution ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5)
    proba = results[best_name]["proba"]
    ax5.hist(proba[y==0], bins=40, alpha=0.6, color=ACCENT2,  label="Stable",        density=True)
    ax5.hist(proba[y==1], bins=40, alpha=0.6, color=WARN,     label="Deteriorating", density=True)
    ax5.axvline(0.30, color="white",  lw=1.5, ls="--", alpha=0.7, label="Low|Medium (0.30)")
    ax5.axvline(0.65, color="#ffa657", lw=1.5, ls="--", alpha=0.7, label="Medium|High (0.65)")
    ax5.set_xlabel("Predicted Risk Score", color=MUTED, fontsize=10)
    ax5.set_ylabel("Density", color=MUTED, fontsize=10)
    ax5.set_title("Risk Score Distribution\nby Outcome", color=TEXT, fontsize=12, fontweight="bold")
    ax5.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, edgecolor="#30363d")

    # ── Metrics comparison bar chart ────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6)
    metric_names = ["AUROC", "AUPRC", "Recall", "Precision", "F1"]
    metric_keys  = ["auroc", "auprc", "recall", "precision", "f1"]
    x = np.arange(len(metric_names))
    w = 0.35
    for i, (name, clr) in enumerate(colors_map.items()):
        vals = [results[name][k] for k in metric_keys]
        bars = ax6.bar(x + i*w - w/2, vals, w, label=name, color=clr, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax6.set_xticks(x); ax6.set_xticklabels(metric_names, color=TEXT, fontsize=10)
    ax6.set_ylim(0, 1.15)
    ax6.set_title("Model Performance Comparison", color=TEXT, fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, edgecolor="#30363d")

    # ── Title ────────────────────────────────────────────────────────────────
    fig.suptitle("ICU Patient Deterioration Risk — Model Evaluation Report",
                 color=TEXT, fontsize=15, fontweight="bold", y=0.98)

    fig.savefig(OUTPUTS_DIR / "evaluation_report.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nEvaluation plots saved to {OUTPUTS_DIR}/evaluation_report.png")


if __name__ == "__main__":
    train_and_evaluate()
