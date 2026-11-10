# Hospital Appointment No-Show Prediction & Scheduling Decision Support

A machine learning system engineered to assist hospital operations and clinic front-desk staff in predicting the probability of missed outpatient appointments. The model functions as a **supportive decision-support tool** to prioritize human-reviewed patient reminders, coordinate transportation assistance, and safely plan scheduling buffers.

> **CRITICAL CLINICAL BOUNDARY**: This system is **administrative only**. It strictly excludes clinical diagnoses, treatments, symptoms, and medical acuity variables. It **never predicts medical diagnoses or health outcomes**, and it must **never be used to automatically cancel, refuse, or deprioritize appointments**.

---

## 1. Problem Formulation & Operational Scope

| Dimension | Specification |
| :--- | :--- |
| **Business Decision** | Allocation of front-desk outreach resources (personalized reminder calls, transit assistance, telehealth options) and conservative buffer planning. |
| **Prediction Unit** | An individual scheduled appointment booking (`appointment_id`). |
| **Target Variable** | Binary `no_show` (`1` = Patient failed to attend without prior cancellation; `0` = Patient attended). |
| **Prediction Horizon** | At appointment booking or operational reminder cutoff ($T-48$ hours prior to scheduled appointment). |
| **Point in Time & Temporal Guardrails** | Strictly uses operational information known at or before prediction time. Historical patient visit statistics are dynamically accumulated chronologically prior to the current appointment. |
| **Cost Matrix & Asymmetry** | **False Negative ($C_{FN} = \$120$)**: Missed appointment results in idle doctor time, empty clinic rooms, and lost access for waitlisted patients.<br>**False Positive ($C_{FP} = \$15$)**: Attending patient receives a 2-3 minute human confirmation phone call.<br>Cost Ratio: **8:1 ($C_{FN}/C_{FP}$)**. |

---

## 2. Architecture & Directory Structure

```
Hospital Appointment No Show Prediction/
├── data/
│   ├── raw/
│   │   └── appointments_synthetic.csv        # 6,000 synthetic operational records
│   └── processed/
│       ├── train.csv                         # 70% chronological split (4,200 rows)
│       ├── val.csv                           # 15% chronological split (900 rows)
│       └── test.csv                          # 15% chronological split (900 rows)
├── artifacts/
│   ├── model_pipeline.joblib                 # Calibrated scikit-learn pipeline artifact
│   ├── model_metadata.json                   # Hyperparameters, leaderboard, threshold metadata
│   ├── eda_report.json                       # Automated data quality & leakage checks
│   ├── evaluation_report.md                  # Detailed performance and governance report
│   └── figures/                              # High-resolution publication charts
│       ├── calibration_curve.png             # Reliability diagram (Platt Sigmoid vs Raw)
│       ├── pr_curve.png                      # Precision-Recall curve with operating thresholds
│       ├── roc_curve.png                     # Receiver Operating Characteristic
│       ├── confusion_matrices.png            # Multi-threshold confusion matrix comparison
│       └── threshold_cost_curve.png          # Operational financial cost curve vs cutoff
├── src/
│   ├── config.py                             # Project paths, features, costs, seed
│   ├── data_generator.py                     # Synthetic data generator with temporal tracking
│   ├── eda.py                                # Data quality checks & leakage detection
│   ├── feature_engineering.py                # Chronological splitting & ColumnTransformer
│   ├── models.py                             # Rule-based baseline + classifier pipelines
│   ├── train.py                              # Model training, cross-validation, calibration
│   └── evaluate.py                           # Plotting & markdown report generation
├── app/
│   ├── api.py                                # FastAPI REST service (/predict, /batch-predict)
│   └── dashboard.py                          # Streamlit interactive operational application
├── tests/
│   ├── test_data.py                          # Synthetic data & temporal validity tests
│   ├── test_pipeline.py                      # Preprocessing & inference pipeline tests
│   └── test_api.py                           # FastAPI route integration tests
├── run_pipeline.py                           # One-click reproducible end-to-end pipeline runner
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## 3. Data Engineering & Leakage Prevention

### Synthetic Data Generation (6,000 Records)
Generates outpatient appointments spanning 10 months with an empirical **no-show rate of ~23.9%** (realistic healthcare baseline):
- **Logistics**: `lead_time_days` (1-60 days), `time_of_day_slot`, `day_of_week`, `booking_channel`.
- **Friction Proxies**: `distance_to_clinic_km` (lognormal distribution), `sms_reminder_opt_in`.
- **Demographic / Administrative**: `age_group`, `insurance_type`, `department`.
- **Strictly Excluded**: ICD-10 diagnostic codes, vitals, prescriptions, clinical notes, and post-appointment data.

### Temporal Leakage Prevention
1. **Dynamic Sequential History**: For every appointment, `prior_appointments_count` and `prior_no_show_rate` are calculated using events occurring strictly *before* that appointment date.
2. **Chronological Train/Val/Test Split**:
   - **Train Set (70%, 4,200 rows)**: Jan 2, 2026 to Jul 2, 2026
   - **Validation Set (15%, 900 rows)**: Jul 2, 2026 to Aug 5, 2026
   - **Test Set (15%, 900 rows)**: Aug 5, 2026 to Oct 26, 2026
   - *Test data represents the unseen operational future. No random cross-temporal shuffling.*

---

## 4. Modeling, Calibration & Cost Optimization

### Validation Leaderboard

| Model Candidate | Val ROC-AUC | Val PR-AUC | Val Brier Score | Val Recall (0.50) | Val Cost (0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Balanced Logistic Regression** | **0.6977** | **0.4591** | **0.2200** | **63.0%** | **$13,245** |
| Baseline Logistic Regression | 0.6978 | 0.4573 | 0.1644 | 21.0% | $21,135 |
| Random Forest Classifier | 0.6796 | 0.4477 | 0.2014 | 47.0% | $16,095 |
| HistGradientBoosting Classifier | 0.6589 | 0.4056 | 0.2136 | 49.8% | $16,125 |
| Heuristic Rule Baseline | 0.6054 | 0.3082 | 0.1977 | 39.7% | $17,745 |

### Probability Calibration
Raw model outputs can produce distorted probabilities. Using **Platt Sigmoid Calibration** fitted via TimeSeries cross-validation:
- **Test Brier Score Before Calibration**: `0.2259`
- **Test Brier Score After Calibration**: **`0.1695`** (25% reduction in calibration error)

### Threshold & Financial Cost Analysis
Because an empty slot ($C_{FN} = \$120$) costs 8x more than a reminder call ($C_{FP} = \$15$), the symmetrical $0.50$ cutoff leaves the hospital vulnerable to missed appointments.

Operating at the **Cost-Optimal Cutoff ($t^* = 0.13$)** on the 900-appointment test cohort:
- **Naive Policy (No Outreach)**: Total Cost = **$28,560** (238 missed slots)
- **Calibrated Model at 0.50 Cutoff**: Total Cost = **$21,975** (Recall = 24.8%, 179 missed slots)
- **Calibrated Model at 0.13 Cutoff**: Total Cost = **$9,705** (Recall = **95.8%**, only 10 missed slots!)
- **Net Cost Savings**: **$18,855 saved vs naive operations** and **$12,270 saved vs standard 0.50 cutoff**.

---

## 5. Quickstart & Reproducibility Guide

### 1. Environment Setup
```powershell
# Create Python 3.12 virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execute Full End-to-End Pipeline
Run data generation, EDA, feature engineering, model training, calibration, and evaluation in one command:
```powershell
python run_pipeline.py
```

### 3. Run Automated Test Suite
```powershell
pytest tests/ -v
```

### 4. Launch FastAPI REST Service
```powershell
uvicorn app.api:app --reload --host 127.0.0.1 --port 8000
```
Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.

### 5. Launch Interactive Streamlit Operations Dashboard
```powershell
streamlit run app/dashboard.py
```
Open `http://localhost:8501` to explore:
- **Live Scheduling (Single Inference)**: Input appointment features and get immediate calibrated probabilities and staff actions.
- **Daily Clinic Roster**: Prioritized worklist for operations staff with call status tracking.
- **Cost Simulator**: Interactive sliders for $C_{FN}$ and $C_{FP}$ to visualize cost curves and optimal thresholds.
- **Performance Hub**: High-res reliability diagrams, PR curves, and confusion matrices.
- **Governance & Ethics**: Human-in-the-loop guidelines and bias mitigation framework.

---

## 6. AI Governance, Ethics & Human-in-the-Loop Safeguards

1. **Supportive, Never Punitive**:
   - A predicted high probability of missing an appointment triggers supportive interventions (personalized phone confirmation, transit vouchers, flexible rescheduling, or same-day telehealth).
   - The model has no automated write access to scheduling databases and cannot drop patients.
2. **Mitigating Demographic & Proxy Bias**:
   - Geographic distance and insurance category correlate with structural socioeconomic friction. Overbooking must be conservative and reviewed by clinical department managers.
3. **Data Minimization & Privacy**:
   - Only operational timing and administrative variables are used. All protected health information (PHI) and clinical diagnostic data are strictly excluded.
4. **Human Review Disclaimers**:
   - Every API prediction response and dashboard screen displays an explicit disclaimer reminding staff of tool limitations and required human confirmation.
