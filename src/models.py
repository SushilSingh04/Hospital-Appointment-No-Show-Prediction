"""Model definitions and factory functions for hospital no-show prediction."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED
from src.feature_engineering import build_preprocessor


class HeuristicRuleBaseline(BaseEstimator, ClassifierMixin):
    """Simple operational rule-of-thumb heuristic baseline used in many hospitals:

    Flag as no-show if:
      - Prior no-show rate >= 0.35 OR
      - Lead time >= 21 days (and no SMS reminder)
    """

    def __init__(self, lead_time_cutoff: int = 21, prior_rate_cutoff: float = 0.35):
        self.lead_time_cutoff = lead_time_cutoff
        self.prior_rate_cutoff = prior_rate_cutoff
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        # If X is DataFrame
        if isinstance(X, pd.DataFrame):
            lead = X["lead_time_days"].values
            prior = X["prior_no_show_rate"].values
            sms = X["sms_reminder_opt_in"].values
        else:
            # Assume first column is lead_time_days, 5th is prior_no_show_rate, etc.
            lead = X[:, 0]
            prior = X[:, 4]
            sms = X[:, 5]

        # Rule evaluation
        flags = ((prior >= self.prior_rate_cutoff) | ((lead >= self.lead_time_cutoff) & (sms == 0))).astype(float)
        # Assign coarse probabilities: 0.70 for flagged, 0.15 for unflagged
        p1 = np.where(flags == 1, 0.70, 0.15)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.50).astype(int)


def get_model_candidates() -> Dict[str, Pipeline]:
    """Instantiate model candidate pipelines combining preprocessing and classifiers."""
    preprocessor = build_preprocessor()

    candidates = {
        "Baseline_LogisticRegression": Pipeline(steps=[
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_SEED,
            )),
        ]),
        "Balanced_LogisticRegression": Pipeline(steps=[
            ("prep", build_preprocessor()),
            ("clf", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
            )),
        ]),
        "RandomForest_Classifier": Pipeline(steps=[
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=180,
                max_depth=9,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting_Classifier": Pipeline(steps=[
            ("prep", build_preprocessor()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.06,
                max_depth=5,
                min_samples_leaf=15,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),
    }

    return candidates
