"""Configuration settings for Hospital Appointment No-Show Prediction."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Ensure directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for reproducibility
RANDOM_SEED = 42

# Prediction Scope & Unit
PREDICTION_UNIT = "Individual Scheduled Appointment (appointment_id)"
TARGET_COLUMN = "no_show"  # 1 = Patient failed to attend, 0 = Attended

# Point in time of prediction
# Prediction is generated at appointment booking / lead-time scheduling window (T-days before appointment)
# ONLY operational variables available at this exact moment are used.

# Guardrails & Ethical Scope:
# NO diagnosis, clinical acuity, treatment, or medical history is included.
# This model is strictly an administrative decision-support tool for staffing, reminders, and buffer management.

IDENTIFIER_COLS = ["appointment_id", "patient_id"]
DATE_COLS = ["booking_datetime", "appointment_datetime"]

NUMERICAL_FEATURES = [
    "lead_time_days",
    "distance_to_clinic_km",
    "prior_appointments_count",
    "prior_no_show_rate",
    "sms_reminder_opt_in",
]

CATEGORICAL_FEATURES = [
    "department",
    "booking_channel",
    "insurance_type",
    "time_of_day_slot",
    "day_of_week",
    "age_group",
]

ALL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Business Costs (USD)
# Cost of False Negative (Clinic slot remains empty, physician idle time, lost access): $120.00
# Cost of False Positive (Cost of staff member placing 2-3 minute human reminder/confirmation call): $15.00
COST_FALSE_NEGATIVE = 120.0
COST_FALSE_POSITIVE = 15.0
COST_TRUE_NEGATIVE = 0.0
COST_TRUE_POSITIVE = 0.0  # Or cost of successful call prevention

# Default risk thresholds
DEFAULT_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.50
MEDIUM_RISK_THRESHOLD = 0.25
