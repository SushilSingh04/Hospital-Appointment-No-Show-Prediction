"""Unit tests for feature engineering, pipeline inference, and calibration."""

import joblib
import numpy as np
import pandas as pd
import pytest
from src.config import ARTIFACTS_DIR
from src.feature_engineering import (
    add_engineered_features,
    chronological_train_val_test_split,
)
from src.train import compute_operational_cost, find_optimal_threshold


def test_engineered_features_creation():
    raw_df = pd.DataFrame([{
        "lead_time_days": 16,
        "distance_to_clinic_km": 18.2,
        "prior_appointments_count": 0,
        "prior_no_show_rate": 0.0,
        "sms_reminder_opt_in": 1,
        "department": "General Medicine",
        "booking_channel": "Online Portal",
        "insurance_type": "Private",
        "time_of_day_slot": "Morning (08:00-11:30)",
        "day_of_week": "Monday",
        "age_group": "31-50",
    }])

    feat_df = add_engineered_features(raw_df)
    assert "log_lead_time" in feat_df.columns
    assert feat_df["is_first_appointment"].iloc[0] == 1
    assert feat_df["high_lead_time_flag"].iloc[0] == 1
    assert feat_df["far_distance_flag"].iloc[0] == 1


def test_chronological_split_strict_ordering():
    df = pd.DataFrame({
        "appointment_datetime": pd.date_range("2026-01-01", periods=100, freq="D"),
        "lead_time_days": np.random.randint(1, 30, size=100),
        "no_show": np.random.randint(0, 2, size=100),
    })

    train, val, test = chronological_train_val_test_split(df, train_ratio=0.7, val_ratio=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15

    # Chronological integrity check
    assert train["appointment_datetime"].max() < val["appointment_datetime"].min()
    assert val["appointment_datetime"].max() < test["appointment_datetime"].min()


def test_loaded_model_pipeline_predicts_valid_probabilities():
    model_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    assert model_path.exists(), "Model artifact must exist"
    model = joblib.load(model_path)

    sample = pd.DataFrame([{
        "appointment_id": "APT_TEST",
        "patient_id": "PAT_TEST",
        "lead_time_days": 10,
        "distance_to_clinic_km": 5.0,
        "prior_appointments_count": 2,
        "prior_no_show_rate": 0.0,
        "sms_reminder_opt_in": 1,
        "department": "General Medicine",
        "booking_channel": "Phone Call",
        "insurance_type": "Private",
        "time_of_day_slot": "Morning (08:00-11:30)",
        "day_of_week": "Wednesday",
        "age_group": "51-65",
    }])
    sample_feat = add_engineered_features(sample)
    probs = model.predict_proba(sample_feat)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
    assert np.isclose(probs[0, 0] + probs[0, 1], 1.0)


def test_cost_function_and_threshold_search():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Model predictions
    y_prob = np.array([0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 0.8, 0.9])

    best_thresh, best_res = find_optimal_threshold(y_true, y_prob)
    assert 0.05 <= best_thresh <= 0.95
    assert "total_cost" in best_res
    assert best_res["tp"] >= 0
