"""Synthetic data generator for Hospital Appointment No-Show Prediction.

Generates realistic appointment operations data strictly avoiding clinical/diagnostic variables.
Ensures point-in-time temporal consistency for historical features.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_DATA_DIR


def generate_synthetic_appointments(
    num_records: int = 6000,
    num_patients: int = 1500,
    start_date: str = "2026-01-01",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate realistic synthetic hospital appointments with temporal integrity.

    Parameters
    ----------
    num_records : int
        Total number of appointments to generate.
    num_patients : int
        Number of distinct patients to simulate repeat visits.
    start_date : str
        Base start date for bookings (YYYY-MM-DD).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic appointments dataset ordered chronologically.
    """
    rng = np.random.default_rng(seed)

    # 1. Base Patient Pool attributes (stable demographic/location proxies, NO clinical labels)
    patient_ids = [f"PAT_{i:05d}" for i in range(1, num_patients + 1)]
    age_groups = ["18-30", "31-50", "51-65", "65+"]
    age_group_probs = [0.22, 0.35, 0.25, 0.18]
    patient_age_group = rng.choice(age_groups, size=num_patients, p=age_group_probs)

    insurance_types = ["Private", "Public / Medicaid", "Medicare", "Self-Pay"]
    insurance_probs = [0.50, 0.25, 0.15, 0.10]
    patient_insurance = rng.choice(insurance_types, size=num_patients, p=insurance_probs)

    # Distance to clinic: lognormal distribution (median ~7 km, max ~45 km)
    patient_distance = np.clip(rng.lognormal(mean=1.9, sigma=0.6, size=num_patients), 0.5, 50.0).round(1)

    # Patient default propensity to opt into SMS
    patient_sms_pref = rng.choice([0, 1], size=num_patients, p=[0.20, 0.80])

    patient_df = pd.DataFrame({
        "patient_id": patient_ids,
        "age_group": patient_age_group,
        "insurance_type": patient_insurance,
        "distance_to_clinic_km": patient_distance,
        "sms_reminder_opt_in": patient_sms_pref,
    })

    # 2. Appointment timeline generation
    base_start = datetime.strptime(start_date, "%Y-%m-%d")
    departments = [
        "General Medicine",
        "Orthopedics",
        "Cardiology Admin",
        "Dermatology",
        "Ophthalmology",
        "ENT",
        "Physical Therapy",
    ]
    dept_weights = [0.30, 0.16, 0.12, 0.14, 0.10, 0.08, 0.10]

    booking_channels = ["Online Portal", "Phone Call", "In-Person Desk", "Provider Referral"]
    channel_weights = [0.45, 0.35, 0.10, 0.10]

    time_slots = ["Morning (08:00-11:30)", "Afternoon (12:00-16:30)", "Evening (17:00-19:30)"]
    time_slot_weights = [0.45, 0.40, 0.15]

    # Assign appointments to patients with geometric-like distribution (some repeat frequent visitors)
    patient_weights = rng.exponential(scale=1.0, size=num_patients)
    patient_weights /= patient_weights.sum()
    assigned_patients = rng.choice(patient_ids, size=num_records, p=patient_weights)

    # Generate booking timestamps across a 240-day window
    booking_offsets_days = rng.uniform(0, 240, size=num_records)
    booking_minutes = rng.integers(8 * 60, 18 * 60, size=num_records)  # 08:00 to 18:00

    # Lead times: Exponential distribution between 1 and 60 days (median ~11 days)
    lead_times_days = np.clip(np.round(rng.exponential(scale=11.0, size=num_records)), 1, 60).astype(int)

    records = []
    for i in range(num_records):
        b_day = base_start + timedelta(days=float(booking_offsets_days[i]))
        b_dt = b_day.replace(hour=int(booking_minutes[i] // 60), minute=int(booking_minutes[i] % 60), second=0)

        # Appointment date is booking date + lead time
        # Hospital clinic operates Monday to Saturday (skip Sunday)
        a_dt = b_dt + timedelta(days=int(lead_times_days[i]))
        if a_dt.weekday() == 6:  # Sunday
            a_dt += timedelta(days=1)
            lead_times_days[i] += 1

        slot = rng.choice(time_slots, p=time_slot_weights)
        if slot.startswith("Morning"):
            hour = rng.integers(8, 12)
        elif slot.startswith("Afternoon"):
            hour = rng.integers(12, 17)
        else:
            hour = rng.integers(17, 20)
        minute = rng.choice([0, 15, 30, 45])
        a_dt = a_dt.replace(hour=int(hour), minute=int(minute), second=0)

        records.append({
            "appointment_id": f"APT_{i+1:06d}",
            "patient_id": assigned_patients[i],
            "booking_datetime": b_dt,
            "appointment_datetime": a_dt,
            "lead_time_days": int(lead_times_days[i]),
            "department": rng.choice(departments, p=dept_weights),
            "booking_channel": rng.choice(booking_channels, p=channel_weights),
            "time_of_day_slot": slot,
            "day_of_week": a_dt.strftime("%A"),
        })

    raw_df = pd.DataFrame(records)
    raw_df = raw_df.merge(patient_df, on="patient_id", how="left")

    # Sort strictly by appointment_datetime to accurately calculate historical past behavior
    raw_df = raw_df.sort_values(by=["appointment_datetime", "booking_datetime"]).reset_index(drop=True)

    # 3. Compute dynamic historical features (strictly prior to each appointment date to prevent leakage)
    # Track patient history sequentially
    patient_history = {pid: {"total": 0, "no_shows": 0} for pid in patient_ids}

    prior_appts = []
    prior_no_shows = []
    prior_rates = []
    no_show_targets = []

    # Pre-calculate base log-odds coefficients
    # Operations research calibrated formula
    base_logit = -2.15  # Baseline ~20-22% no-show rate

    for idx, row in raw_df.iterrows():
        pid = row["patient_id"]
        hist = patient_history[pid]
        p_total = hist["total"]
        p_ns = hist["no_shows"]
        p_rate = (p_ns / p_total) if p_total > 0 else 0.0

        prior_appts.append(p_total)
        prior_no_shows.append(p_ns)
        prior_rates.append(round(p_rate, 3))

        # Calculate latent log-odds of no-show for this appointment
        logit = base_logit

        # Lead time effect (forgetting / rescheduling friction)
        logit += 0.042 * row["lead_time_days"]

        # Prior no-show history effect (habitual behavior)
        if p_total > 0:
            logit += 1.85 * p_rate
        else:
            logit += 0.10  # First-time patients have slight uncertainty

        # Age group effect
        if row["age_group"] == "18-30":
            logit += 0.38
        elif row["age_group"] == "31-50":
            logit += 0.12
        elif row["age_group"] == "51-65":
            logit -= 0.18
        elif row["age_group"] == "65+":
            logit -= 0.48

        # Insurance type effect (proxy for healthcare navigation/transportation challenges)
        if row["insurance_type"] == "Self-Pay":
            logit += 0.35
        elif row["insurance_type"] == "Public / Medicaid":
            logit += 0.28
        elif row["insurance_type"] == "Private":
            logit -= 0.15

        # Distance friction
        logit += 0.018 * min(row["distance_to_clinic_km"], 40.0)

        # SMS reminder opt-in
        if row["sms_reminder_opt_in"] == 1:
            logit -= 0.42

        # Booking channel
        if row["booking_channel"] == "Online Portal":
            logit += 0.22
        elif row["booking_channel"] == "Provider Referral":
            logit -= 0.32
        elif row["booking_channel"] == "In-Person Desk":
            logit -= 0.20

        # Day of week
        if row["day_of_week"] in ["Monday", "Friday"]:
            logit += 0.20
        elif row["day_of_week"] == "Saturday":
            logit += 0.28

        # Time slot
        if "Evening" in row["time_of_day_slot"]:
            logit += 0.25
        elif "Morning" in row["time_of_day_slot"]:
            logit -= 0.10

        # Department baseline variance
        if row["department"] == "Physical Therapy":
            logit += 0.20
        elif row["department"] == "Cardiology Admin":
            logit -= 0.30
        elif row["department"] == "Ophthalmology":
            logit -= 0.15

        # Compute probability via sigmoid
        prob = 1.0 / (1.0 + np.exp(-logit))
        # Bernoulli trial
        is_no_show = int(rng.random() < prob)
        no_show_targets.append(is_no_show)

        # Update history for this patient AFTER recording this appointment
        patient_history[pid]["total"] += 1
        patient_history[pid]["no_shows"] += is_no_show

    raw_df["prior_appointments_count"] = prior_appts
    raw_df["prior_no_shows_count"] = prior_no_shows
    raw_df["prior_no_show_rate"] = prior_rates
    raw_df["no_show"] = no_show_targets

    # Format datetime strings
    raw_df["booking_datetime"] = pd.to_datetime(raw_df["booking_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    raw_df["appointment_datetime"] = pd.to_datetime(raw_df["appointment_datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    return raw_df


def save_synthetic_data(output_path: Optional[str] = None) -> pd.DataFrame:
    """Generate and persist the synthetic hospital appointments dataset."""
    df = generate_synthetic_appointments()
    out = RAW_DATA_DIR / "appointments_synthetic.csv" if output_path is None else output_path
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} records to {out}")
    print(f"Overall No-Show Rate: {df['no_show'].mean():.2%}")
    return df


if __name__ == "__main__":
    save_synthetic_data()
