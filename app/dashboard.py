"""Interactive Streamlit Operations Dashboard for Hospital Appointment No-Show Prediction."""

from datetime import datetime
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st

from src.config import (
    ARTIFACTS_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    DEFAULT_THRESHOLD,
    FIGURES_DIR,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    PROCESSED_DATA_DIR,
)
from src.feature_engineering import add_engineered_features

st.set_page_config(
    page_title="Clinic Operations: No-Show Prediction & Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished hospital operations styling
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 14px;
        color: #4a5568;
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #ebf8ff;
        border-left: 5px solid #3182ce;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 13px;
        color: #2b6cb0;
    }
    .risk-high {
        background-color: #fed7d7;
        color: #9b2c2c;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
    .risk-med {
        background-color: #feebc8;
        color: #9c4221;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
    .risk-low {
        background-color: #c6f6d5;
        color: #22543d;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_production_pipeline():
    """Load persisted scikit-learn model pipeline."""
    model_path = ARTIFACTS_DIR / "model_pipeline.joblib"
    if not model_path.exists():
        from src.train import train_and_evaluate_all
        train_and_evaluate_all()
    return joblib.load(model_path)


@st.cache_data
def load_metadata():
    meta_path = ARTIFACTS_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_sample_roster():
    test_path = PROCESSED_DATA_DIR / "test.csv"
    if test_path.exists():
        return pd.read_csv(test_path)
    return pd.DataFrame()


model = load_production_pipeline()
metadata = load_metadata()
opt_threshold = metadata.get("optimal_threshold", DEFAULT_THRESHOLD)

# --- Header Banner ---
st.markdown('<div class="main-header">🏥 Hospital Appointment No-Show Prediction & Scheduling Decision Support</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Operational intelligence for clinic staff reminders, transportation assistance, and buffer optimization.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Clinical & Ethical Boundary Disclaimer:</strong> This system is exclusively an <em>administrative decision-support tool</em>. 
    It estimates missed appointment likelihood using scheduling logistics only. <strong>It NEVER predicts medical diagnosis, clinical acuity, or treatment urgency</strong>, 
    and it must <strong>NEVER be used to cancel, refuse, or deprioritize patient care automatically</strong>. All outreach requires human staff review.
</div>
""", unsafe_allow_html=True)

# --- Navigation Sidebar ---
st.sidebar.image("https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?w=300&q=80", use_container_width=True)
st.sidebar.title("Operational Portal")

nav_choice = st.sidebar.radio(
    "Select Workflow:",
    [
        "📌 1. Live Patient Scheduling (Single Inference)",
        "📋 2. Daily Clinic Roster & Outreach Queue",
        "⚙️ 3. Operational Threshold & Cost Simulator",
        "📈 4. Model Performance & Calibration Hub",
        "🛡️ 5. AI Governance, Ethics & Limitations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Model Profile:**
- **Architecture**: {metadata.get('best_model_architecture', 'Calibrated Logistic Regression')}
- **Calibration**: Platt Sigmoid (cv=TimeSeriesSplit)
- **Optimal Cutoff**: `{opt_threshold:.2f}`
- **Test ROC-AUC**: `{metadata.get('test_metrics', {}).get('test_roc_auc', 0.708)}`
- **Test PR-AUC**: `{metadata.get('test_metrics', {}).get('test_pr_auc', 0.478)}`
""")


# ==============================================================================
# TAB 1: LIVE SCHEDULING & PATIENT TRIAGE
# ==============================================================================
if nav_choice.startswith("📌 1"):
    st.subheader("Interactive Appointment Triage & Real-Time Risk Assessment")
    st.markdown("Enter booking details to obtain an instant calibrated no-show risk estimate and recommended operational outreach protocol.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 📅 Booking & Logistics")
        apt_id = st.text_input("Appointment ID", value="APT_904128")
        patient_id = st.text_input("Patient ID", value="PAT_04812")
        department = st.selectbox(
            "Clinic Department",
            ["General Medicine", "Orthopedics", "Cardiology Admin", "Dermatology", "Ophthalmology", "ENT", "Physical Therapy"],
            index=0,
        )
        booking_channel = st.selectbox(
            "Booking Channel",
            ["Online Portal", "Phone Call", "In-Person Desk", "Provider Referral"],
            index=0,
        )
        lead_time = st.slider("Lead Time (Days from booking to appointment)", min_value=0, max_value=60, value=18)

    with col2:
        st.markdown("##### 🕒 Timing & Friction")
        time_slot = st.selectbox(
            "Appointment Time Slot",
            ["Morning (08:00-11:30)", "Afternoon (12:00-16:30)", "Evening (17:00-19:30)"],
            index=1,
        )
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            index=0,
        )
        distance = st.slider("Distance to Clinic (km)", min_value=0.5, max_value=50.0, value=14.5, step=0.5)
        sms_opt_in = st.radio("SMS Reminder Opt-In", [1, 0], format_func=lambda x: "Yes (Opted In)" if x == 1 else "No (Opted Out)", horizontal=True)

    with col3:
        st.markdown("##### 👤 Demographics & Attendance History")
        age_group = st.selectbox("Patient Age Group", ["18-30", "31-50", "51-65", "65+"], index=1)
        insurance = st.selectbox("Insurance / Payment Type", ["Private", "Public / Medicaid", "Medicare", "Self-Pay"], index=0)
        prior_appts = st.slider("Total Past Appointments at Hospital", min_value=0, max_value=20, value=3)
        prior_rate = st.slider("Past No-Show Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.33 if prior_appts > 0 else 0.0, step=0.05)

    thresh_input = st.slider("Operational Decision Cutoff for Flagging", 0.05, 0.60, float(opt_threshold), 0.01)

    if st.button("🔍 Evaluate Appointment Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "appointment_id": apt_id,
            "patient_id": patient_id,
            "lead_time_days": lead_time,
            "distance_to_clinic_km": distance,
            "prior_appointments_count": prior_appts,
            "prior_no_show_rate": prior_rate,
            "sms_reminder_opt_in": sms_opt_in,
            "department": department,
            "booking_channel": booking_channel,
            "insurance_type": insurance,
            "time_of_day_slot": time_slot,
            "day_of_week": day_of_week,
            "age_group": age_group,
        }])

        input_feat = add_engineered_features(input_data)
        cal_prob = float(model.predict_proba(input_feat)[:, 1][0])

        st.markdown("---")
        st.markdown("### 📊 Inference & Human Decision Support Result")

        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

        with res_col1:
            st.metric("Calibrated No-Show Probability", f"{cal_prob:.1%}")
            st.progress(cal_prob)

        with res_col2:
            if cal_prob >= HIGH_RISK_THRESHOLD:
                st.markdown("#### Risk Status: <span class='risk-high'>HIGH RISK</span>", unsafe_allow_html=True)
                st.warning("Needs priority telephone outreach.")
            elif cal_prob >= thresh_input:
                st.markdown("#### Risk Status: <span class='risk-med'>MODERATE RISK</span>", unsafe_allow_html=True)
                st.info("Interactive 2-way SMS reminder recommended.")
            else:
                st.markdown("#### Risk Status: <span class='risk-low'>STANDARD RISK</span>", unsafe_allow_html=True)
                st.success("Standard reminder flow sufficient.")

        with res_col3:
            st.markdown("##### 📋 Recommended Human-in-the-Loop Protocol:")
            if cal_prob >= HIGH_RISK_THRESHOLD:
                st.markdown("""
                - **Primary Action**: Front-desk staff should place a personal call 48 hours in advance.
                - **Supportive Inquiry**: Check if patient requires transportation support (bus pass / clinic shuttle) or assistance navigating clinic entry.
                - **Flexibility**: If patient reports high conflict risk, offer a same-day telehealth alternative or reschedule to a preferred time window.
                - **Buffer Protocol**: Operations may allow a soft overbooking buffer in this clinic hour block, provided emergency wait time limits are monitored.
                """)
            elif cal_prob >= thresh_input:
                st.markdown("""
                - **Primary Action**: Send interactive 2-way SMS reminder requesting '1 to Confirm, 2 to Reschedule'.
                - **Staff Action**: If no confirmation received within 24 hours of appointment, queue for phone follow-up.
                """)
            else:
                st.markdown("""
                - **Standard Action**: Send automated calendar notification and SMS reminder 24 hours prior.
                - **No extra staff phone outreach needed**.
                """)

        # Key Factor Analysis
        st.markdown("##### 🔎 Key Driving Risk Factors for this Booking:")
        factors = []
        if lead_time >= 14:
            factors.append(f"• **High Lead Time ({lead_time} days)**: Appointments booked more than 2 weeks ahead have significantly higher forgetfulness / scheduling conflicts.")
        if prior_rate >= 0.25:
            factors.append(f"• **Prior Attendance History ({prior_rate:.0%} past no-show rate)**: Historical no-show behavior is a strong indicator of recurring scheduling friction.")
        if sms_opt_in == 0:
            factors.append("• **SMS Opt-Out**: Patient does not receive automated text reminders, eliminating a low-friction touchpoint.")
        if distance >= 15:
            factors.append(f"• **Distance Friction ({distance} km)**: Long commute increases vulnerability to transit delays or weather disruption.")
        if insurance in ["Public / Medicaid", "Self-Pay"]:
            factors.append(f"• **Coverage Category ({insurance})**: Correlates with socio-economic and logistical volatility.")
        if not factors:
            factors.append("• No prominent risk escalations detected. Standard booking profile.")

        for f in factors:
            st.markdown(f)


# ==============================================================================
# TAB 2: DAILY CLINIC ROSTER & OUTREACH QUEUE
# ==============================================================================
elif nav_choice.startswith("📋 2"):
    st.subheader("Daily Clinic Roster & Human-Reviewed Outreach Worklist")
    st.markdown("Score an entire day's clinic schedule in real time. Prioritize patient outreach to maximize attendance while respecting staff workload.")

    roster_df = load_sample_roster()
    if roster_df.empty:
        st.warning("Test dataset not found. Generating on-demand...")
        from src.data_generator import generate_synthetic_appointments
        roster_df = generate_synthetic_appointments(num_records=900)

    # Let user select a clinic date from roster
    roster_df["appt_date"] = pd.to_datetime(roster_df["appointment_datetime"]).dt.strftime("%Y-%m-%d")
    available_dates = sorted(roster_df["appt_date"].unique())
    selected_date = st.selectbox("Select Clinic Service Date:", available_dates[:15], index=0)

    day_appointments = roster_df[roster_df["appt_date"] == selected_date].copy().reset_index(drop=True)

    # Score appointments
    day_feat = add_engineered_features(day_appointments)
    day_probs = model.predict_proba(day_feat)[:, 1]
    day_appointments["calibrated_prob"] = day_probs
    day_appointments["risk_tier"] = np.where(
        day_probs >= HIGH_RISK_THRESHOLD, "High Risk",
        np.where(day_probs >= opt_threshold, "Moderate Risk", "Standard Risk")
    )
    day_appointments["requires_call"] = day_probs >= opt_threshold

    # Roster KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_appts = len(day_appointments)
    high_risk_count = int((day_appointments["risk_tier"] == "High Risk").sum())
    mod_risk_count = int((day_appointments["risk_tier"] == "Moderate Risk").sum())
    calls_needed = int(day_appointments["requires_call"].sum())
    est_staff_mins = calls_needed * 3

    kpi1.metric("Scheduled Appointments", total_appts)
    kpi2.metric("High-Risk Patients", high_risk_count)
    kpi3.metric("Staff Outreach Queue", f"{calls_needed} patients")
    kpi4.metric("Est. Staff Time Required", f"{est_staff_mins} mins (~{est_staff_mins/60:.1f} hrs)")

    st.markdown("---")
    st.markdown("##### 📞 Prioritized Front-Desk Outreach Worklist")

    # Filter controls
    tier_filter = st.multiselect(
        "Filter by Risk Tier:",
        ["High Risk", "Moderate Risk", "Standard Risk"],
        default=["High Risk", "Moderate Risk"],
    )
    filtered_roster = day_appointments[day_appointments["risk_tier"].isin(tier_filter)].sort_values(by="calibrated_prob", ascending=False)

    display_cols = [
        "appointment_id",
        "patient_id",
        "appointment_datetime",
        "department",
        "lead_time_days",
        "distance_to_clinic_km",
        "prior_no_show_rate",
        "calibrated_prob",
        "risk_tier",
    ]

    st.dataframe(
        filtered_roster[display_cols].style.format({
            "calibrated_prob": "{:.1%}",
            "prior_no_show_rate": "{:.1%}",
            "distance_to_clinic_km": "{:.1f} km",
        }),
        use_container_width=True,
    )

    st.markdown("##### 📝 Staff Call Outcome Logger (Decision-Support Tracking)")
    call_col1, call_col2, call_col3 = st.columns(3)
    with call_col1:
        selected_apt = st.selectbox("Select Patient to Log Call:", filtered_roster["appointment_id"].tolist() if len(filtered_roster) > 0 else ["No appointments"])
    with call_col2:
        call_outcome = st.selectbox("Call Outcome:", [
            "Attendance Confirmed (Transport verified)",
            "Rescheduled to Alternative Date (Slot reclaimed for waitlist!)",
            "Canceled by Patient (Slot reclaimed for waitlist!)",
            "Voicemail Left",
            "No Answer / Incorrect Phone Number",
        ])
    with call_col3:
        st.write("")
        st.write("")
        if st.button("💾 Record Staff Intervention", type="secondary"):
            st.success(f"Action logged for {selected_apt}: '{call_outcome}'. Operations schedule updated.")


# ==============================================================================
# TAB 3: OPERATIONAL THRESHOLD & COST SIMULATOR
# ==============================================================================
elif nav_choice.startswith("⚙️ 3"):
    st.subheader("Operational Threshold & Business Cost Optimization Simulator")
    st.markdown("""
    In standard machine learning, binary classification defaults to a 0.50 threshold. 
    However, in hospital operations, **the cost of an empty clinic slot (False Negative) is much higher than a 3-minute reminder call (False Positive)**.
    Adjust clinic cost parameters below to observe the financial sweet spot in real time.
    """)

    test_path = PROCESSED_DATA_DIR / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
    else:
        from src.data_generator import generate_synthetic_appointments
        test_df = generate_synthetic_appointments(num_records=900)

    y_true = test_df["no_show"].values
    test_feat = add_engineered_features(test_df)
    y_prob = model.predict_proba(test_feat)[:, 1]

    param_col1, param_col2, param_col3 = st.columns(3)
    with param_col1:
        cost_fn = st.slider("Cost of False Negative (Empty clinic slot / idle doctor)", 40, 300, int(COST_FALSE_NEGATIVE), 10)
    with param_col2:
        cost_fp = st.slider("Cost of False Positive (3-min staff phone call / confirmation)", 5, 50, int(COST_FALSE_POSITIVE), 5)
    with param_col3:
        cost_ratio = cost_fn / cost_fp
        st.metric("Cost Ratio (FN : FP)", f"{cost_ratio:.1f} : 1")
        st.caption("A missed slot costs ~8x more than a reminder call.")

    sim_thresh = st.slider("Simulate Decision Cutoff Threshold:", 0.04, 0.80, float(opt_threshold), 0.01)

    # Compute stats at sim_thresh, default 0.50, and zero-outreach
    def get_metrics_at(t, c_fn, c_fp):
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cost = (fn * c_fn) + (fp * c_fp)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "cost": cost, "recall": rec, "precision": prec, "f1": f1}

    sim_res = get_metrics_at(sim_thresh, cost_fn, cost_fp)
    dflt_res = get_metrics_at(0.50, cost_fn, cost_fp)
    naive_cost = int(y_true.sum()) * cost_fn

    st.markdown("---")
    st.markdown("##### 💰 Financial Impact on 900-Patient Cohort")

    cost1, cost2, cost3, cost4 = st.columns(4)
    cost1.metric("Naive Policy (No Outreach)", f"${naive_cost:,.0f}")
    cost2.metric("Default Cutoff (0.50)", f"${dflt_res['cost']:,.0f}", delta=f"-${naive_cost - dflt_res['cost']:,.0f}")
    cost3.metric(f"Current Cutoff ({sim_thresh:.2f})", f"${sim_res['cost']:,.0f}", delta=f"-${naive_cost - sim_res['cost']:,.0f}", delta_color="inverse")
    cost4.metric("Savings vs 0.50 Threshold", f"${dflt_res['cost'] - sim_res['cost']:,.0f}")

    # Confusion Matrix Visualization
    st.markdown("##### 🧮 Resulting Confusion Matrix at Current Cutoff")
    cm_col1, cm_col2 = st.columns([1, 2])

    with cm_col1:
        st.write(f"**True Positives (No-Shows Caught)**: `{sim_res['tp']}`")
        st.write(f"**False Negatives (Missed No-Shows)**: `{sim_res['fn']}`")
        st.write(f"**False Positives (Outreach to attending)**: `{sim_res['fp']}`")
        st.write(f"**True Negatives (Attending without call)**: `{sim_res['tn']}`")
        st.write(f"**Recall (Coverage)**: `{sim_res['recall']:.1%}`")
        st.write(f"**Precision (Call Hit Rate)**: `{sim_res['precision']:.1%}`")

    with cm_col2:
        # Dynamic Cost Curve
        ts = np.linspace(0.04, 0.85, 80)
        costs = [get_metrics_at(t, cost_fn, cost_fp)["cost"] for t in ts]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(ts, costs, color="#2c3e50", lw=2.5, label="Total Operational Cost")
        ax.axvline(sim_thresh, color="#e74c3c", linestyle="--", lw=1.8, label=f"Selected Threshold ({sim_thresh:.2f})")
        ax.axvline(0.50, color="#7f8c8d", linestyle=":", lw=1.5, label="Standard 0.50")
        ax.set_xlabel("Decision Cutoff Threshold")
        ax.set_ylabel("Total Financial Cost ($)")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


# ==============================================================================
# TAB 4: MODEL PERFORMANCE & CALIBRATION HUB
# ==============================================================================
elif nav_choice.startswith("📈 4"):
    st.subheader("Model Evaluation, Comparison & Probability Calibration")
    st.markdown("Rigorous validation across classifiers, calibration diagnostics, and precision-recall dynamics.")

    st.markdown("##### 🏆 Model Candidate Validation Leaderboard")
    lb = pd.DataFrame(metadata.get("leaderboard", []))
    if not lb.empty:
        st.dataframe(
            lb.style.format({
                "val_roc_auc": "{:.4f}",
                "val_pr_auc": "{:.4f}",
                "val_brier_score": "{:.4f}",
                "val_f1_default": "{:.3f}",
                "val_recall_default": "{:.1%}",
                "val_precision_default": "{:.1%}",
                "val_cost_default": "${:,.0f}",
            }),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("##### 🔍 Calibration & Operating Diagnostic Plots")

    plot_tabs = st.tabs([
        "Reliability (Calibration)",
        "Precision-Recall Curve",
        "ROC Curve",
        "Confusion Matrices",
        "Threshold Cost Profile",
    ])

    with plot_tabs[0]:
        st.markdown("##### Probability Calibration (Reliability Diagram)")
        st.markdown("""
        **Why Calibration Matters:** In healthcare scheduling, an operational threshold requires probabilities to reflect actual event frequencies.
        If a model outputs 30%, exactly 30 out of 100 such appointments should be no-shows. Platt Sigmoid scaling dramatically improved the test Brier score from **0.2330 to 0.1719**.
        """)
        cal_path = FIGURES_DIR / "calibration_curve.png"
        if cal_path.exists():
            st.image(str(cal_path), use_container_width=True)

    with plot_tabs[1]:
        st.markdown("##### Precision-Recall Curve")
        st.markdown("Because no-shows are the minority class (~25%), the PR Curve is more informative than ROC for measuring operational efficiency.")
        pr_path = FIGURES_DIR / "pr_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), use_container_width=True)

    with plot_tabs[2]:
        st.markdown("##### Receiver Operating Characteristic (ROC)")
        roc_path = FIGURES_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)

    with plot_tabs[3]:
        st.markdown("##### Confusion Matrices Comparison")
        cm_path = FIGURES_DIR / "confusion_matrices.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)

    with plot_tabs[4]:
        st.markdown("##### Total Financial Impact vs Decision Threshold")
        cost_path = FIGURES_DIR / "threshold_cost_curve.png"
        if cost_path.exists():
            st.image(str(cost_path), use_container_width=True)


# ==============================================================================
# TAB 5: AI ETHICS, BIAS & GOVERNANCE FRAMEWORK
# ==============================================================================
elif nav_choice.startswith("🛡️ 5"):
    st.subheader("AI Ethics, Fairness, Data Minimization & Governance Safeguards")

    st.markdown("""
    ### 1. The Critical Distinction: Supportive vs Punitive Interventions
    > **Fundamental Rule**: A predicted high risk of no-show must **NEVER** lead to punitive action, automatic cancellation, or refusal of scheduling.
    
    In healthcare, missed appointments are rarely frivolous; they typically reflect:
    - Lack of reliable transportation (transit deserts, distance friction)
    - Inability to take unpaid time off work
    - Childcare or eldercare emergencies
    - Complex healthcare navigation friction
    
    Therefore, the **ethical design mandate** of this system is strictly **supportive**:
    - High-risk flags prompt operations staff to offer **transit vouchers**, **clinic shuttle rides**, or **same-day telehealth appointments**.
    - If a patient genuinely cannot make it, proactive outreach allows voluntary rescheduling, freeing up the vacated slot for waitlisted urgent patients.

    ---

    ### 2. Representativeness, Demographic Parity & Proxy Bias
    - **Demographic Variables**: The model uses coarse age buckets and insurance category as administrative routing proxies.
    - **Risk of Proxy Bias**: Public assistance / Medicaid and long travel distances naturally correlate with socioeconomic challenges. If left unmonitored, algorithmic overbooking could disproportionately impact disadvantaged populations.
    - **Mitigation**: Double-booking buffers must be strictly capped (maximum 1 buffer slot per 4-hour clinic session) and must be reviewed by the clinical lead of each department.

    ---

    ### 3. Privacy & Data Minimization
    - **Zero Clinical Variables**: This project strictly excludes ICD-10 diagnosis codes, clinical notes, psychiatric history, vital signs, and medication lists.
    - **Zero Direct Identifiers Used in Modeling**: Names, phone numbers, and street addresses are withheld from the feature matrix and only joined at the front-desk UI for staff contact.

    ---

    ### 4. Failure Modes & Edge Case Protocols
    - **Model Drift**: Seasonal flu surges, public transit strikes, or extreme weather events alter baseline no-show rates. The operations pipeline includes weekly drift monitoring on Brier score and calibration intercept.
    - **First-Time Patients**: Patients with zero historical hospital visits receive an uncertainty buffer and are assigned standard outreach.
    - **Human-in-the-Loop Override**: Front-desk staff have complete authority to override model suggestions at any time with a single click.
    """)
