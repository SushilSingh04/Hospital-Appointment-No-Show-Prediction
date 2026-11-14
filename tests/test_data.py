"""Unit tests for synthetic data generation and data quality checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest
from src.config import TARGET_COLUMN, DATE_COLS
from src.data_generator import generate_synthetic_appointments
from src.eda import run_data_quality_and_eda


def test_synthetic_data_generation_shape_and_types():
    df = generate_synthetic_appointments(num_records=1200, num_patients=300, seed=123)
    assert len(df) == 1200
    assert df["patient_id"].nunique() <= 300
    assert TARGET_COLUMN in df.columns
    assert df[TARGET_COLUMN].isin([0, 1]).all()


def test_no_clinical_variables_leakage():
    df = generate_synthetic_appointments(num_records=500, num_patients=100, seed=456)
    prohibited_terms = ["diag", "treat", "medication", "acuity", "vitals", "prescription", "symptom"]
    for col in df.columns:
        for term in prohibited_terms:
            assert term not in col.lower(), f"Prohibited clinical term '{term}' found in column '{col}'"


def test_temporal_validity():
    df = generate_synthetic_appointments(num_records=800, num_patients=200, seed=789)
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col])

    # Appointment must not occur before booking
    assert (df["appointment_datetime"] >= df["booking_datetime"]).all()
    # Lead time days must be non-negative
    assert (df["lead_time_days"] >= 0).all()


def test_target_class_balance_within_healthcare_bounds():
    df = generate_synthetic_appointments(num_records=1500, num_patients=400, seed=999)
    positive_rate = df[TARGET_COLUMN].mean()
    # Realistic healthcare outpatient no-show rate is between 15% and 35%
    assert 0.15 <= positive_rate <= 0.35, f"Unexpected no-show rate: {positive_rate:.2%}"


def test_eda_health_check_passes():
    report = run_data_quality_and_eda()
    assert report["leakage_analysis"]["leakage_detected"] is False
    assert report["duplicate_appointments"] == 0
    assert report["temporal_inconsistencies"] == 0
