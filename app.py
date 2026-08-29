import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fund Allocation Dashboard", layout="wide")

st.title("🩺 AI-Based Health Screening & Fund Allocation Dashboard")

# Load data
df = pd.read_csv('/content/drive/MyDrive/eyelid_dataset/phase5_fund_allocation_results.csv')

# ---- Summary Metrics ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", len(df))
col2.metric("Suspicious Cases", (df['anomaly_label'] == 'Suspicious').sum())
col3.metric("Total Allocated", f"₹{df['allocated_amount'].sum():,.0f}")
col4.metric("High Risk Cases", (df['risk_label'] == 'High').sum())

st.divider()

# ---- Filters ----
st.sidebar.header("Filters")
risk_filter = st.sidebar.multiselect("Risk Level", options=df['risk_label'].unique(), default=df['risk_label'].unique())
anomaly_filter = st.sidebar.multiselect("Anomaly Status", options=df['anomaly_label'].unique(), default=df['anomaly_label'].unique())

filtered_df = df[(df['risk_label'].isin(risk_filter)) & (df['anomaly_label'].isin(anomaly_filter))]

# ---- Charts ----
c1, c2 = st.columns(2)
with c1:
    st.subheader("Risk Level Distribution")
    st.bar_chart(df['risk_label'].value_counts())

with c2:
    st.subheader("Normal vs Suspicious")
    st.bar_chart(df['anomaly_label'].value_counts())

st.subheader("Top Priority Cases")
st.dataframe(
    filtered_df.sort_values('priority_score', ascending=False)[
        ['image_name', 'risk_label', 'anomaly_label', 'priority_score', 'requested_amount', 'allocated_amount']
    ],
    use_container_width=True
)

st.divider()
st.subheader("⚠️ Suspicious Cases (Held for Manual Review)")
st.dataframe(
    df[df['anomaly_label'] == 'Suspicious'][
        ['image_name', 'risk_label', 'requested_amount', 'num_previous_requests']
    ],
    use_container_width=True
)

st.divider()
st.header("🆕 New Case Entry — Live Risk Check")

import joblib
import numpy as np

# Load saved models
risk_model = joblib.load('/content/drive/MyDrive/eyelid_dataset/phase3_risk_model.pkl')
anomaly_model = joblib.load('/content/drive/MyDrive/eyelid_dataset/phase4_isolation_forest.pkl')

with st.form("new_case_form"):
    col1, col2 = st.columns(2)
    with col1:
        new_age = st.number_input("Age", min_value=1, max_value=100, value=30)
        new_requested_amount = st.number_input("Requested Amount (₹)", min_value=0, value=5000)
        new_num_previous = st.number_input("Number of Previous Requests", min_value=0, value=0)
    with col2:
        new_symptom_score = st.number_input("Symptom Score", min_value=0, max_value=5, value=1)
        new_risk_score = st.selectbox("Risk Score (from Phase 2 CV model output)", options=[0, 1, 2], format_func=lambda x: {0:"Low", 1:"Medium", 2:"High"}[x])

    submitted = st.form_submit_button("Check Case")

    if submitted:
        # Anomaly check using same 5 features as training
        input_features = np.array([[new_age, new_risk_score, new_requested_amount, new_num_previous, new_symptom_score]])
        anomaly_result = anomaly_model.predict(input_features)[0]
        anomaly_status = "Suspicious" if anomaly_result == -1 else "Normal"

        risk_label_map = {0: "Low", 1: "Medium", 2: "High"}
        risk_label = risk_label_map[new_risk_score]

        # Priority score (same formula as Phase 5)
        priority_score = (new_risk_score * 3) + new_symptom_score
        if anomaly_status == "Suspicious":
            priority_score -= 5

        st.subheader("Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Risk Level", risk_label)
        r2.metric("Anomaly Status", anomaly_status)
        r3.metric("Priority Score", priority_score)

        if anomaly_status == "Suspicious":
            st.error("⚠️ This case is flagged SUSPICIOUS — held for manual review, fund not auto-allocated.")
        else:
            st.success(f"✅ Normal case — eligible for fund allocation based on priority score {priority_score}.")
