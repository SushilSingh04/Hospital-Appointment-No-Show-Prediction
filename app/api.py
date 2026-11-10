"""FastAPI application exposing real-time and batch appointment no-show predictions."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
)
from src.feature_engineering import add_engineered_features

# Global model state
MODEL = None
METADATA = {}


def load_model_and_metadata():
    """Load model artifact and metadata safely."""
    global MODEL, METADATA
    model_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    meta_path = ARTIFACTS_DIR / "model_metadata.json"

    if model_path.exists():
        MODEL = joblib.load(model_path)
    if meta_path.exists():
        with open(meta_path, "r") as f:
            METADATA = json.load(f)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    load_model_and_metadata()
    yield


app = FastAPI(
    title="Hospital Appointment No-Show Prediction API",
    description="Operational decision-support tool for clinic staffing, buffer management, and human-in-the-loop reminders. Not for clinical diagnosis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class AppointmentItem(BaseModel):
    appointment_id: str = Field(..., examples=["APT_100234"])
    patient_id: str = Field(..., examples=["PAT_00452"])
    lead_time_days: int = Field(..., ge=0, le=180, examples=[18])
    distance_to_clinic_km: float = Field(..., ge=0.0, le=200.0, examples=[12.5])
    prior_appointments_count: int = Field(..., ge=0, examples=[4])
    prior_no_show_rate: float = Field(..., ge=0.0, le=1.0, examples=[0.25])
    sms_reminder_opt_in: int = Field(..., ge=0, le=1, examples=[1])
    department: str = Field(..., examples=["General Medicine"])
    booking_channel: str = Field(..., examples=["Online Portal"])
    insurance_type: str = Field(..., examples=["Private"])
    time_of_day_slot: str = Field(..., examples=["Morning (08:00-11:30)"])
    day_of_week: str = Field(..., examples=["Monday"])
    age_group: str = Field(..., examples=["31-50"])


class BatchAppointmentRequest(BaseModel):
    appointments: List[AppointmentItem]
    custom_threshold: Optional[float] = Field(None, ge=0.01, le=0.99)


class PredictionResponse(BaseModel):
    appointment_id: str
    patient_id: str
    calibrated_no_show_probability: float
    predicted_no_show: int
    decision_threshold_used: float
    risk_tier: str
    recommended_action: str
    disclaimer: str


class BatchPredictionResponse(BaseModel):
    total_processed: int
    flagged_for_outreach: int
    risk_tier_counts: Dict[str, int]
    predictions: List[PredictionResponse]


# --- Helper Logic ---
def classify_risk_tier_and_action(prob: float, threshold: float) -> Tuple[str, str]:
    """Categorize probability into operational risk tiers and actionable protocols."""
    if prob >= HIGH_RISK_THRESHOLD:
        tier = "High Risk"
        action = "Priority 1: Front-desk phone confirmation call within 48h; offer transit support or telehealth alternative if friction identified."
    elif prob >= threshold:
        tier = "Moderate Risk"
        action = "Priority 2: Interactive 2-way SMS reminder; flag for backup clinic buffer if slot remains unconfirmed."
    else:
        tier = "Standard / Low Risk"
        action = "Standard Protocol: Automated SMS appointment reminder 24h prior."
    return tier, action


# --- API Routes ---
@app.get("/health", tags=["System"])
def health_check():
    """Verify API availability and model status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": MODEL is not None,
        "model_architecture": METADATA.get("best_model_architecture", "Unknown"),
        "optimal_threshold": METADATA.get("optimal_threshold", DEFAULT_THRESHOLD),
    }


@app.get("/metrics", tags=["Model Analytics"])
def get_metrics():
    """Return model evaluation leaderboard and test set performance."""
    if not METADATA:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model metadata not loaded.")
    return METADATA


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(appointment: AppointmentItem, threshold: Optional[float] = None):
    """Predict calibrated no-show probability for an individual appointment."""
    if MODEL is None:
        load_model_and_metadata()
        if MODEL is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact unavailable.")

    operating_threshold = threshold if threshold is not None else METADATA.get("optimal_threshold", DEFAULT_THRESHOLD)

    # Convert single item to DataFrame
    df = pd.DataFrame([appointment.model_dump()])
    df_feat = add_engineered_features(df)

    prob = float(MODEL.predict_proba(df_feat)[:, 1][0])
    is_no_show = int(prob >= operating_threshold)
    risk_tier, action = classify_risk_tier_and_action(prob, operating_threshold)

    return PredictionResponse(
        appointment_id=appointment.appointment_id,
        patient_id=appointment.patient_id,
        calibrated_no_show_probability=round(prob, 4),
        predicted_no_show=is_no_show,
        decision_threshold_used=round(operating_threshold, 3),
        risk_tier=risk_tier,
        recommended_action=action,
        disclaimer="DECISION SUPPORT ONLY: This model informs administrative reminders. It must never be used to automatically drop bookings or prioritize medical diagnoses.",
    )


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchAppointmentRequest):
    """Predict no-show probabilities for a batch of scheduled appointments."""
    if MODEL is None:
        load_model_and_metadata()
        if MODEL is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact unavailable.")

    operating_threshold = (
        payload.custom_threshold
        if payload.custom_threshold is not None
        else METADATA.get("optimal_threshold", DEFAULT_THRESHOLD)
    )

    items = [appt.model_dump() for appt in payload.appointments]
    df = pd.DataFrame(items)
    df_feat = add_engineered_features(df)

    probs = MODEL.predict_proba(df_feat)[:, 1]

    results: List[PredictionResponse] = []
    risk_counts = {"High Risk": 0, "Moderate Risk": 0, "Standard / Low Risk": 0}
    flagged_count = 0

    for i, appt in enumerate(payload.appointments):
        p = float(probs[i])
        pred = int(p >= operating_threshold)
        tier, action = classify_risk_tier_and_action(p, operating_threshold)

        risk_counts[tier] = risk_counts.get(tier, 0) + 1
        if pred == 1:
            flagged_count += 1

        results.append(PredictionResponse(
            appointment_id=appt.appointment_id,
            patient_id=appt.patient_id,
            calibrated_no_show_probability=round(p, 4),
            predicted_no_show=pred,
            decision_threshold_used=round(operating_threshold, 3),
            risk_tier=tier,
            recommended_action=action,
            disclaimer="DECISION SUPPORT ONLY: For operational staff reminders and scheduling buffer planning.",
        ))

    return BatchPredictionResponse(
        total_processed=len(results),
        flagged_for_outreach=flagged_count,
        risk_tier_counts=risk_counts,
        predictions=results,
    )
