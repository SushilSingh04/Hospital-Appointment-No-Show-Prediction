"""Training, calibration, model selection, and artifact generation pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    ARTIFACTS_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    DEFAULT_THRESHOLD,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from src.feature_engineering import add_engineered_features, prepare_and_save_splits
from src.models import HeuristicRuleBaseline, get_model_candidates


def compute_operational_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> Dict[str, Any]:
    """Compute business cost, confusion matrix components, and metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_cost = (fn * cost_fn) + (fp * cost_fp)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": round(threshold, 3),
        "total_cost": round(float(total_cost), 2),
        "cost_fn_component": round(float(fn * cost_fn), 2),
        "cost_fp_component": round(float(fp * cost_fp), 2),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.10, 0.90, 81),
) -> Tuple[float, Dict[str, Any]]:
    """Scan thresholds to locate the minimum operational cost point."""
    best_cost = float("inf")
    best_res = None
    best_thresh = 0.50

    for t in thresholds:
        res = compute_operational_cost(y_true, y_prob, threshold=t)
        if res["total_cost"] < best_cost:
            best_cost = res["total_cost"]
            best_res = res
            best_thresh = t

    return best_thresh, best_res


def train_and_evaluate_all() -> Dict[str, Any]:
    """Execute end-to-end model training, comparison, calibration, and artifact generation."""
    train_path = PROCESSED_DATA_DIR / "train.csv"
    val_path = PROCESSED_DATA_DIR / "val.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        train_df, val_df, test_df = prepare_and_save_splits()
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

    y_train = train_df[TARGET_COLUMN].values
    y_val = val_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # Evaluate Heuristic Baseline
    heuristic = HeuristicRuleBaseline()
    val_prob_heuristic = heuristic.predict_proba(val_df)[:, 1]
    test_prob_heuristic = heuristic.predict_proba(test_df)[:, 1]

    val_res_heuristic = compute_operational_cost(y_val, val_prob_heuristic, threshold=0.50)
    test_res_heuristic = compute_operational_cost(y_test, test_prob_heuristic, threshold=0.50)

    model_candidates = get_model_candidates()
    leaderboard = []

    trained_pipelines = {}
    val_predictions = {"Heuristic_Rule_Baseline": val_prob_heuristic}
    test_predictions = {"Heuristic_Rule_Baseline": test_prob_heuristic}

    leaderboard.append({
        "model_name": "Heuristic_Rule_Baseline",
        "val_roc_auc": round(float(roc_auc_score(y_val, val_prob_heuristic)), 4),
        "val_pr_auc": round(float(average_precision_score(y_val, val_prob_heuristic)), 4),
        "val_brier_score": round(float(brier_score_loss(y_val, val_prob_heuristic)), 4),
        "val_f1_default": val_res_heuristic["f1"],
        "val_precision_default": val_res_heuristic["precision"],
        "val_recall_default": val_res_heuristic["recall"],
        "val_cost_default": val_res_heuristic["total_cost"],
    })

    for name, pipe in model_candidates.items():
        print(f"Training candidate model: {name}...")
        pipe.fit(train_df, y_train)
        trained_pipelines[name] = pipe

        val_prob = pipe.predict_proba(val_df)[:, 1]
        test_prob = pipe.predict_proba(test_df)[:, 1]

        val_predictions[name] = val_prob
        test_predictions[name] = test_prob

        cost_res_default = compute_operational_cost(y_val, val_prob, threshold=0.50)

        leaderboard.append({
            "model_name": name,
            "val_roc_auc": round(float(roc_auc_score(y_val, val_prob)), 4),
            "val_pr_auc": round(float(average_precision_score(y_val, val_prob)), 4),
            "val_brier_score": round(float(brier_score_loss(y_val, val_prob)), 4),
            "val_f1_default": cost_res_default["f1"],
            "val_precision_default": cost_res_default["precision"],
            "val_recall_default": cost_res_default["recall"],
            "val_cost_default": cost_res_default["total_cost"],
        })

    leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="val_pr_auc", ascending=False)
    print("\n--- Model Validation Leaderboard ---")
    print(leaderboard_df.to_string(index=False))

    # Pick top performing model based on PR-AUC (best metric for imbalanced positive class)
    best_candidate_name = leaderboard_df.iloc[0]["model_name"]
    best_raw_pipe = trained_pipelines[best_candidate_name]
    print(f"\nTop Model Selected: {best_candidate_name}")

    # Calibration using Sigmoid/Platt scaling via TimeSeries cross-validation on the training set
    # Preserves temporal integrity across cross-validation folds
    fresh_candidate = get_model_candidates()[best_candidate_name]
    calibrated_clf = CalibratedClassifierCV(
        estimator=fresh_candidate,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    calibrated_clf.fit(train_df, y_train)

    # Evaluate calibration improvement
    raw_test_prob = best_raw_pipe.predict_proba(test_df)[:, 1]
    cal_test_prob = calibrated_clf.predict_proba(test_df)[:, 1]

    raw_brier = brier_score_loss(y_test, raw_test_prob)
    cal_brier = brier_score_loss(y_test, cal_test_prob)
    print(f"Test Brier Score Before Calibration: {raw_brier:.4f}")
    print(f"Test Brier Score After Calibration:  {cal_brier:.4f}")

    # Optimal threshold search on validation split (using calibrated probabilities on val)
    cal_val_prob = calibrated_clf.predict_proba(val_df)[:, 1]
    optimal_thresh, val_opt_stats = find_optimal_threshold(y_val, cal_val_prob)
    print(f"Cost-Optimal Decision Threshold (found on Val): {optimal_thresh:.2f}")

    # Out-of-sample Test Evaluation at default vs optimal threshold
    test_eval_default = compute_operational_cost(y_test, cal_test_prob, threshold=0.50)
    test_eval_optimal = compute_operational_cost(y_test, cal_test_prob, threshold=optimal_thresh)
    test_eval_baseline = test_res_heuristic

    # Zero-intervention cost benchmark (predict show for everyone: FN on all actual no-shows)
    total_actual_no_shows = int(y_test.sum())
    naive_no_intervention_cost = total_actual_no_shows * COST_FALSE_NEGATIVE

    test_summary = {
        "selected_model": best_candidate_name,
        "is_calibrated": True,
        "optimal_threshold": round(float(optimal_thresh), 3),
        "test_roc_auc": round(float(roc_auc_score(y_test, cal_test_prob)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, cal_test_prob)), 4),
        "test_brier_score": round(float(cal_brier), 4),
        "eval_at_optimal_threshold": test_eval_optimal,
        "eval_at_default_threshold": test_eval_default,
        "eval_heuristic_baseline": test_eval_baseline,
        "naive_no_intervention_cost": round(float(naive_no_intervention_cost), 2),
        "savings_vs_no_intervention_usd": round(float(naive_no_intervention_cost - test_eval_optimal["total_cost"]), 2),
        "savings_vs_default_threshold_usd": round(float(test_eval_default["total_cost"] - test_eval_optimal["total_cost"]), 2),
    }

    print("\n--- Final Test Set Evaluation ---")
    print(f"ROC-AUC: {test_summary['test_roc_auc']}")
    print(f"PR-AUC:  {test_summary['test_pr_auc']}")
    print(f"At Default (0.50): F1={test_eval_default['f1']}, Recall={test_eval_default['recall']}, Cost=${test_eval_default['total_cost']}")
    print(f"At Optimal ({optimal_thresh:.2f}): F1={test_eval_optimal['f1']}, Recall={test_eval_optimal['recall']}, Cost=${test_eval_optimal['total_cost']}")
    print(f"Cost Savings vs Naive No-Intervention: ${test_summary['savings_vs_no_intervention_usd']}")
    print(f"Cost Savings vs 0.50 Threshold:        ${test_summary['savings_vs_default_threshold_usd']}")

    # Save artifacts
    model_artifact_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    joblib.dump(calibrated_clf, model_artifact_path)

    metadata = {
        "best_model_architecture": best_candidate_name,
        "optimal_threshold": round(float(optimal_thresh), 3),
        "cost_assumptions": {
            "cost_false_negative_usd": COST_FALSE_NEGATIVE,
            "cost_false_positive_usd": COST_FALSE_POSITIVE,
        },
        "leaderboard": leaderboard,
        "test_metrics": test_summary,
    }

    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save test set predictions array for plotting
    np.savez(
        ARTIFACTS_DIR / "eval_predictions.npz",
        y_test=y_test,
        cal_test_prob=cal_test_prob,
        raw_test_prob=raw_test_prob,
        heuristic_prob=test_prob_heuristic,
    )

    print(f"\nArtifacts successfully persisted to {ARTIFACTS_DIR}")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_all()
