# Hospital Appointment No-Show Prediction: Technical & Operational Report

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
| **Heuristic_Rule_Baseline** | 0.6054 | 0.3082 | 0.1977 | 0.4018 | 0.3973 | $17,745 |
| **Baseline_LogisticRegression** | 0.6978 | 0.4573 | 0.1644 | 0.3172 | 0.21 | $21,135 |
| **Balanced_LogisticRegression** | 0.6977 | 0.4591 | 0.22 | 0.4662 | 0.6301 | $13,245 |
| **RandomForest_Classifier** | 0.6796 | 0.4477 | 0.2014 | 0.4411 | 0.4703 | $16,095 |
| **GradientBoosting_Classifier** | 0.6589 | 0.4056 | 0.2136 | 0.4168 | 0.4977 | $16,125 |

### Probability Calibration
Raw classifiers often suffer from miscalibrated confidence scores. Using Platt Sigmoid calibration fitted via TimeSeries cross-validation:
- **Test Brier Score Before Calibration**: 0.2306
- **Test Brier Score After Calibration**: **0.1695** (sharp improvement in probability reliability)

---

## 3. Business Cost of False Predictions & Threshold Analysis

### Cost Matrix Formulation
- **Cost of False Negative ($C_{FN} = \$120$)**: An unpredicted no-show leaves the clinic room and physician idle. The slot cannot be offered to urgent waitlisted patients, causing lost hospital revenue and reduced healthcare access.
- **Cost of False Positive ($C_{FP} = \$15$)**: An attended appointment incorrectly flagged as a no-show triggers a staff member to spend 2-3 minutes placing a personalized confirmation call or verifying transport.
- **Cost Ratio**: $\approx 8:1$ ($C_{FN} / C_{FP}$). Because an empty clinic slot is ~8 times more expensive than a reminder call, operating at the symmetric default threshold ($0.50$) is economically sub-optimal.

### Out-of-Sample Test Comparison (900 Appointments)

| Operating Configuration | Decision Cutoff | Precision | Recall | F1 Score | Missed No-Shows (FN) | Outreach Calls (FP) | Total Financial Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Policy (No Outreach)** | N/A | 0.0% | 0.0% | 0.000 | 231 | 0 | **$28,560** |
| **Heuristic Rule Baseline** | 0.50 | 39.9% | 43.3% | 0.415 | 135 | 155 | **$18,525** |
| **Calibrated Model (Default)** | 0.50 | 64.1% | 24.8% | 0.358 | 179 | 33 | **$21,975** |
| **Calibrated Model (Cost-Optimal)** | **0.13** | **28.7%** | **95.8%** | **0.441** | **10** | **567** | **$9,705** |

### Net Impact
- **Cost Reduction vs Naive Operations**: **$18,855 saved** across 900 test appointments.
- **Cost Reduction vs Default 0.50 Cutoff**: **$12,270 saved**, lifting recall from 24.8% to **95.8%**.

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
