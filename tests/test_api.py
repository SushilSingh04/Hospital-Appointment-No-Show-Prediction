"""Integration tests for FastAPI endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from app.api import (
    health_check,
    get_metrics,
    predict_single,
    predict_batch,
    AppointmentItem,
    BatchAppointmentRequest,
    load_model_and_metadata,
)


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    load_model_and_metadata()


def test_health_endpoint():
    data = health_check()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "optimal_threshold" in data


def test_metrics_endpoint():
    data = get_metrics()
    assert "leaderboard" in data
    assert "test_metrics" in data


def test_predict_single_endpoint():
    payload = AppointmentItem(
        appointment_id="APT_TEST_99",
        patient_id="PAT_TEST_01",
        lead_time_days=21,
        distance_to_clinic_km=15.5,
        prior_appointments_count=3,
        prior_no_show_rate=0.33,
        sms_reminder_opt_in=0,
        department="General Medicine",
        booking_channel="Online Portal",
        insurance_type="Public / Medicaid",
        time_of_day_slot="Evening (17:00-19:30)",
        day_of_week="Friday",
        age_group="18-30",
    )
    res = predict_single(payload)
    assert res.appointment_id == "APT_TEST_99"
    assert 0.0 <= res.calibrated_no_show_probability <= 1.0
    assert res.predicted_no_show in [0, 1]
    assert res.risk_tier in ["High Risk", "Moderate Risk", "Standard / Low Risk"]
    assert "DECISION SUPPORT ONLY" in res.disclaimer


def test_batch_predict_endpoint():
    payload = BatchAppointmentRequest(
        appointments=[
            AppointmentItem(
                appointment_id="APT_001",
                patient_id="PAT_001",
                lead_time_days=2,
                distance_to_clinic_km=3.0,
                prior_appointments_count=5,
                prior_no_show_rate=0.0,
                sms_reminder_opt_in=1,
                department="Orthopedics",
                booking_channel="Phone Call",
                insurance_type="Private",
                time_of_day_slot="Morning (08:00-11:30)",
                day_of_week="Tuesday",
                age_group="65+",
            ),
            AppointmentItem(
                appointment_id="APT_002",
                patient_id="PAT_002",
                lead_time_days=35,
                distance_to_clinic_km=28.0,
                prior_appointments_count=1,
                prior_no_show_rate=1.0,
                sms_reminder_opt_in=0,
                department="General Medicine",
                booking_channel="Online Portal",
                insurance_type="Self-Pay",
                time_of_day_slot="Evening (17:00-19:30)",
                day_of_week="Friday",
                age_group="18-30",
            ),
        ],
        custom_threshold=0.20,
    )
    res = predict_batch(payload)
    assert res.total_processed == 2
    assert len(res.predictions) == 2
    assert "High Risk" in res.risk_tier_counts
