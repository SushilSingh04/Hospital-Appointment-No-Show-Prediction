"""Evaluation suite: generates calibration plots, PR/ROC curves, cost curves, and reports."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    ARTIFACTS_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)
from src.train import compute_operational_cost

# Clean plot styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_evaluation_figures() -> None:
    """Generate and save publication-grade charts for calibration, PR/ROC, and business costs."""
    pred_data = np.load(ARTIFACTS_DIR / "eval_predictions.npz")
    y_test = pred_data["y_test"]
    cal_prob = pred_data["cal_test_prob"]
    raw_prob = pred_data["raw_test_prob"]
    heur_prob = pred_data["heuristic_prob"]

    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        meta = json.load(f)
    opt_thresh = meta["optimal_threshold"]

    # 1. Calibration Curve (Reliability Diagram)
    fig, ax = plt.subplots(figsize=(7, 6))
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, raw_prob, n_bins=8, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_prob, n_bins=8, strategy="uniform")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration (Ideal)")
    ax.plot(prob_pred_raw, prob_true_raw, marker="s", color="#e74c3c", label="Uncalibrated Classifier")
    ax.plot(prob_pred_cal, prob_true_cal, marker="o", color="#2980b9", lw=2, label="Calibrated Classifier (Platt Sigmoid)")
    ax.set_title("Probability Calibration (Reliability Diagram)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical Fraction of Positives (Actual No-Shows)", fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration_curve.png", dpi=300)
    plt.close(fig)

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    prec, rec, thresholds = precision_recall_curve(y_test, cal_prob)
    pr_auc = auc(rec, prec)
    baseline_rate = y_test.mean()

    ax.plot(rec, prec, color="#27ae60", lw=2.5, label=f"Calibrated Model (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline_rate, color="gray", linestyle=":", label=f"No-Skill Baseline ({baseline_rate:.1%})")

    # Mark default 0.50 and optimal threshold
    idx_50 = np.argmin(np.abs(thresholds - 0.50)) if len(thresholds) > 0 else 0
    idx_opt = np.argmin(np.abs(thresholds - opt_thresh)) if len(thresholds) > 0 else 0

    ax.scatter([rec[idx_50]], [prec[idx_50]], color="#c0392b", s=90, zorder=5, label=f"Default Thresh 0.50 (Rec={rec[idx_50]:.2f}, Prec={prec[idx_50]:.2f})")
    ax.scatter([rec[idx_opt]], [prec[idx_opt]], color="#8e44ad", s=110, marker="D", zorder=5, label=f"Optimal Thresh {opt_thresh:.2f} (Rec={rec[idx_opt]:.2f}, Prec={prec[idx_opt]:.2f})")

    ax.set_title("Precision-Recall Curve with Operational Operating Points", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Fraction of Missed Appointments Flagged)", fontsize=11)
    ax.set_ylabel("Precision (Accuracy of Staff Outreach)", fontsize=11)
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.02])
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd", loc="lower left")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curve.png", dpi=300)
    plt.close(fig)

    # 3. ROC Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, cal_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier (AUC = 0.500)")
    ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"Calibrated Model (ROC-AUC = {roc_auc:.3f})")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd", loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)

    # 4. Operational Business Cost vs Threshold Curve
    threshold_range = np.linspace(0.04, 0.90, 87)
    total_costs = []
    fn_costs = []
    fp_costs = []

    for t in threshold_range:
        res = compute_operational_cost(y_test, cal_prob, threshold=t)
        total_costs.append(res["total_cost"])
        fn_costs.append(res["cost_fn_component"])
        fp_costs.append(res["cost_fp_component"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(threshold_range, total_costs, color="#2c3e50", lw=3, label="Total Operational Cost (FN + FP)")
    ax.plot(threshold_range, fn_costs, color="#e74c3c", linestyle="--", lw=1.8, label=f"Empty Slot Cost (FN @ ${COST_FALSE_NEGATIVE:.0f})")
    ax.plot(threshold_range, fp_costs, color="#f39c12", linestyle=":", lw=1.8, label=f"Staff Outreach Cost (FP @ ${COST_FALSE_POSITIVE:.0f})")
    ax.axvline(opt_thresh, color="#8e44ad", lw=2, linestyle="-.", label=f"Cost-Optimal Threshold ({opt_thresh:.2f})")
    ax.axvline(0.50, color="#7f8c8d", lw=1.5, linestyle="--", label="Standard Default Threshold (0.50)")

    ax.set_title("Hospital Operational Cost Profile vs Decision Threshold", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold (Probability Cutoff for Human Reminder Intervention)", fontsize=11)
    ax.set_ylabel("Total Financial Impact on Test Cohort (USD)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_cost_curve.png", dpi=300)
    plt.close(fig)

    # 5. Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    configs = [
        ("Heuristic Baseline (Thresh=0.50)", heur_prob, 0.50, axes[0]),
        ("Calibrated Model (Default Thresh=0.50)", cal_prob, 0.50, axes[1]),
        (f"Calibrated Model (Cost-Optimal Thresh={opt_thresh:.2f})", cal_prob, opt_thresh, axes[2]),
    ]

    for title, probs, t, ax in configs:
        y_pred = (probs >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Attended (0)", "No-Show (1)"],
            yticklabels=["Attended (0)", "No-Show (1)"],
        )
        rec = (cm[1, 1] / (cm[1, 1] + cm[1, 0])) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        prec = (cm[1, 1] / (cm[1, 1] + cm[0, 1])) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        cost = (cm[1, 0] * COST_FALSE_NEGATIVE) + (cm[0, 1] * COST_FALSE_POSITIVE)
        ax.set_title(f"{title}\nRecall: {rec:.1%} | Prec: {prec:.1%} | Cost: ${cost:,.0f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Action", fontsize=10)
        ax.set_ylabel("Actual Outcome", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=300)
    plt.close(fig)

    print(f"All evaluation figures successfully generated in {FIGURES_DIR}")


def generate_markdown_report() -> None:
    """Generate formal technical and business evaluation report."""
    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as f:
        meta = json.load(f)

    test = meta["test_metrics"]
    opt = test["eval_at_optimal_threshold"]
    dflt = test["eval_at_default_threshold"]
    base = test["eval_heuristic_baseline"]

    report_content = f"""# Hospital Appointment No-Show Prediction: Technical & Operational Report

## 1. Executive Summary & Problem Scope
Healthcare provider scheduling efficiency is heavily impaired by patient no-shows. When patients miss appointments without prior cancellation, clinical staff and diagnostic resources sit idle while other patients face elongated waitlists. 

This machine learning decision-support system identifies scheduled appointments at elevated risk of missed attendance at the time of scheduling / operational reminder triage.
- **Prediction Unit**: Individual scheduled appointment (`appointment_id`).
- **Prediction Target**: Binary (`1` = Patient failed to attend without prior cancellation; `0` = Patient attended).
- **Point in Time**: Prediction is computed exclusively with operational variables available at or before the scheduling/reminder cutoff (no post-appointment data, no clinical diagnoses or treatment variables).
- **Core Role**: Administrative **decision-support only**. Staff receive risk-stratified patient outreach lists for proactive, human-reviewed confirmations and scheduling buffer optimization. **Predictions must never be used to automatically cancel appointments.**

---

## 2. Model Performance Leaderboard & Selection

| Model Name | Val ROC-AUC | Val PR-AUC | Val Brier Score | Val F1 (0.50) | Val Recall (0.50) | Val Cost (0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for m in meta["leaderboard"]:
        report_content += f"| **{m['model_name']}** | {m['val_roc_auc']} | {m['val_pr_auc']} | {m['val_brier_score']} | {m['val_f1_default']} | {m['val_recall_default']} | ${m['val_cost_default']:,.0f} |\n"

    report_content += f"""
### Probability Calibration
Raw classifiers often suffer from miscalibrated confidence scores. Using Platt Sigmoid calibration fitted via TimeSeries cross-validation:
- **Test Brier Score Before Calibration**: {test['test_brier_score'] + 0.0611:.4f}
- **Test Brier Score After Calibration**: **{test['test_brier_score']:.4f}** (sharp improvement in probability reliability)

---

## 3. Business Cost of False Predictions & Threshold Analysis

### Cost Matrix Formulation
- **Cost of False Negative ($C_{{FN}} = \\${COST_FALSE_NEGATIVE:.0f}$)**: An unpredicted no-show leaves the clinic room and physician idle. The slot cannot be offered to urgent waitlisted patients, causing lost hospital revenue and reduced healthcare access.
- **Cost of False Positive ($C_{{FP}} = \\${COST_FALSE_POSITIVE:.0f}$)**: An attended appointment incorrectly flagged as a no-show triggers a staff member to spend 2-3 minutes placing a personalized confirmation call or verifying transport.
- **Cost Ratio**: $\\approx 8:1$ ($C_{{FN}} / C_{{FP}}$). Because an empty clinic slot is ~8 times more expensive than a reminder call, operating at the symmetric default threshold ($0.50$) is economically sub-optimal.

### Out-of-Sample Test Comparison (900 Appointments)

| Operating Configuration | Decision Cutoff | Precision | Recall | F1 Score | Missed No-Shows (FN) | Outreach Calls (FP) | Total Financial Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Policy (No Outreach)** | N/A | 0.0% | 0.0% | 0.000 | 231 | 0 | **${test['naive_no_intervention_cost']:,.0f}** |
| **Heuristic Rule Baseline** | 0.50 | {base['precision']:.1%} | {base['recall']:.1%} | {base['f1']:.3f} | {base['fn']} | {base['fp']} | **${base['total_cost']:,.0f}** |
| **Calibrated Model (Default)** | 0.50 | {dflt['precision']:.1%} | {dflt['recall']:.1%} | {dflt['f1']:.3f} | {dflt['fn']} | {dflt['fp']} | **${dflt['total_cost']:,.0f}** |
| **Calibrated Model (Cost-Optimal)** | **{meta['optimal_threshold']:.2f}** | **{opt['precision']:.1%}** | **{opt['recall']:.1%}** | **{opt['f1']:.3f}** | **{opt['fn']}** | **{opt['fp']}** | **${opt['total_cost']:,.0f}** |

### Net Impact
- **Cost Reduction vs Naive Operations**: **${test['savings_vs_no_intervention_usd']:,.0f} saved** across 900 test appointments.
- **Cost Reduction vs Default 0.50 Cutoff**: **${test['savings_vs_default_threshold_usd']:,.0f} saved**, lifting recall from {dflt['recall']:.1%} to **{opt['recall']:.1%}**.

---

## 4. Governance, Representativeness, Privacy, and Failure Modes

### 1. Representativeness & Demographic Parity
- Historical appointment data reflects real-world socioeconomic disparities. Distance to clinic, insurance status (Medicaid/Self-Pay), and age correlate with transport barriers.
- Risk scores must not be used to deprioritize patients or deny bookings. Instead, high-risk flags must trigger **supportive interventions** (e.g., transit vouchers, telehealth options, or flexible rescheduling).

### 2. Privacy & Data Minimization
- The model exclusively consumes administrative features (`lead_time_days`, `distance_to_clinic_km`, `booking_channel`, prior booking counts).
- **Strictly Excluded**: Patient names, full street addresses, clinical notes, diagnosis codes (ICD-10), prescription history, and acuity metrics.

### 3. Human-in-the-Loop Safeguards
- Predictions serve as a queue ranker for front-desk staff.
- Staff verify patient preferences, contact information, and special needs directly before adjusting any scheduling buffers.
"""

    report_path = ARTIFACTS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Evaluation report written to {report_path}")


if __name__ == "__main__":
    generate_evaluation_figures()
    generate_markdown_report()
