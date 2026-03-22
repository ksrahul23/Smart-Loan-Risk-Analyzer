import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import sys
import os

# Add root to sys.path for internal imports
root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

try:
    from src.predict import load_pipeline, predict_risk
except ImportError:
    st.error("Could not import prediction modules. Ensure the project structure is correct.")

st.set_page_config(
    page_title="Smart Loan Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
/* Base Theme */
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{
    background:#0f172a !important; /* Deep Navy Dark Mode */
    color:#f8fafc !important;
}
.block-container{
    padding-top:2.4rem !important;
    padding-bottom:1rem !important;
    padding-left:1.1rem !important;
    padding-right:1.1rem !important;
}
h1,h2,h3,h4,p,label,span,div{
    color:#f8fafc !important;
}
.title-text{
    font-size:2rem;
    font-weight:800;
    color:#38bdf8 !important; /* Sky Blue accent */
    margin-bottom:0.15rem;
}
.subtitle-text{
    color:#94a3b8 !important;
    font-size:0.95rem;
    margin-bottom:1rem;
}
.section-title{
    font-size:1.05rem;
    font-weight:700;
    color:#38bdf8 !important;
    margin-bottom:0.8rem;
}
.result-good{
    color:#4ade80 !important;
    font-size:2rem;
    font-weight:800;
    text-align:center;
}
.result-bad{
    color:#f87171 !important;
    font-size:2rem;
    font-weight:800;
    text-align:center;
}
.result-neutral{
    color:#60a5fa !important;
    font-size:2rem;
    font-weight:800;
    text-align:center;
}
.result-sub{
    text-align:center;
    font-size:1rem;
    color:#94a3b8 !important;
}
/* Card Styling */
[data-testid="stVerticalBlockBorderWrapper"]{
    background:#1e293b !important;
    border:1px solid #334155 !important;
    border-radius:18px !important;
    padding:1rem !important;
    box-shadow:0 10px 25px rgba(0,0,0,0.3) !important;
}
[data-testid="stMetric"]{
    background:#1e293b !important;
    border:1px solid #334155 !important;
    border-radius:16px !important;
    padding:0.8rem !important;
    box-shadow:0 6px 18px rgba(15,23,42,0.06) !important;
}
/* Inputs */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div{
    background:#334155 !important;
    border:1px solid #475569 !important;
    border-radius:10px !important;
}
div[data-baseweb="select"] *, div[data-baseweb="input"] *{
    color:#f8fafc !important;
}
/* Submit Button - Solid Teal */
div[data-testid="stFormSubmitButton"] button,
button[kind="primary"]{
    background:#14b8a6 !important;
    color:#ffffff !important;
    border:none !important;
    border-radius:12px !important;
    font-weight:700 !important;
    min-height:2.8rem !important;
    transition:all 0.25s ease !important;
}
div[data-testid="stFormSubmitButton"] button:hover,
button[kind="primary"]:hover{
    background:#0d9488 !important;
    transform:scale(1.02);
}
/* Plotly styles */
.js-plotly-plot .plotly .modebar{
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# 1. Load Model and Metrics
@st.cache_resource
def get_model_and_metrics():
    model_path = root_path / "models" / "loan_model.pkl"
    metrics_path = root_path / "models" / "metrics.json"
    
    pipeline = joblib.load(model_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return pipeline, metrics

pipeline, metrics = get_model_and_metrics()

# 2. State Management
if "prediction_label" not in st.session_state:
    st.session_state.prediction_label = "Waiting for input"
    st.session_state.probability = 0.00
    st.session_state.probability_text = "Probability"
    st.session_state.recommendation = "Fill applicant details and click Predict Risk."
    st.session_state.result_class = "result-neutral"

st.markdown('<div class="title-text">Smart Loan Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Production-grade ML dashboard for predicting loan approval risk.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.0, 1.25, 1.1], gap="large")

# 3. Sidebar/Inputs
with col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Applicant Details</div>', unsafe_allow_html=True)
        with st.form("loan_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", min_value=0, value=1500)
            loan_amount = st.number_input("Loan Amount ($K)", min_value=0, value=120)
            loan_amount_term = st.number_input("Loan Amount Term (Days)", min_value=0, value=360)
            credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x == 1.0 else "Poor/No History")
            property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
            predict_btn = st.form_submit_button("Predict Risk", use_container_width=True)

if predict_btn:
    # Create input DataFrame
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])
    
    pred, app_prob, risk_prob = predict_risk(input_df, pipeline)
    
    # Update Session State
    if pred == 1:
        st.session_state.prediction_label = "LOW RISK"
        st.session_state.probability = app_prob
        st.session_state.probability_text = "Approval Probability"
        st.session_state.recommendation = "Recommended Action: System identifies high likelihood of repayment."
        st.session_state.result_class = "result-good"
    else:
        st.session_state.prediction_label = "HIGH RISK"
        st.session_state.probability = risk_prob
        st.session_state.probability_text = "Rejection Probability"
        st.session_state.recommendation = "Recommended Action: Manual verification required due to risk factors."
        st.session_state.result_class = "result-bad"

# 4. Results & Visuals
with col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Risk Prediction</div>', unsafe_allow_html=True)
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.probability * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#14b8a6"},
                "steps": [
                    {"range": [0, 50], "color": "#e2e8f0"},
                    {"range": [50, 80], "color": "#fef08a"},
                    {"range": [80, 100], "color": "#fecaca"}
                ]
            }
        ))
        gauge_fig.update_layout(height=230, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
        
        st.markdown(f'<div class="{st.session_state.result_class}">{st.session_state.prediction_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-sub">{st.session_state.probability_text}: {st.session_state.probability:.1%}</div>', unsafe_allow_html=True)
        st.progress(st.session_state.probability)
        st.markdown(f'<p class="small-text" style="text-align:center; font-size: 0.9rem; color: #94a3b8; margin-top:10px;">{st.session_state.recommendation}</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">Model Feature Importance</div>', unsafe_allow_html=True)
        
        # Calculate feature importance from pipeline
        model = pipeline.named_steps['classifier']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get feature names from preprocessor
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
        cat_names = cat_encoder.get_feature_names_out(cat_features)
        num_names = preprocessor.named_transformers_['num'].feature_names_in_
        feature_names = list(num_names) + list(cat_names)
        
        importances = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True).tail(8)
        
        fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h", template="plotly_dark")
        fig.update_traces(marker_color="#38bdf8")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# 5. Model Performance Metrics (Dynamic)
with col3:
    m1, m2 = st.columns(2)
    m1.metric("Overall Accuracy", f"{metrics['accuracy']:.0%}")
    m2.metric("ROC-AUC Score", f"{metrics['auc']:.2f}")

    with st.container(border=True):
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = np.array(metrics['confusion_matrix'])
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Reject", "Approve"], yticklabels=["Actual N", "Actual Y"], ax=ax_cm)
        ax_cm.set_title("Validation Set Evaluation", color="white")
        ax_cm.tick_params(colors='white')
        fig_cm.patch.set_facecolor("#1e293b")
        st.pyplot(fig_cm, use_container_width=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">ROC Curve Performance</div>', unsafe_allow_html=True)
        roc_data = metrics['roc_curve']
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3.5))
        ax_roc.plot(roc_data['fpr'], roc_data['tpr'], label=f"RF (AUC={metrics['auc']})", color='#38bdf8', lw=2)
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color='#94a3b8')
        ax_roc.set_xlabel("False Positive Rate", color="white")
        ax_roc.set_ylabel("True Positive Rate", color="white")
        ax_roc.tick_params(colors='white')
        ax_roc.legend()
        fig_roc.patch.set_facecolor("#1e293b")
        st.pyplot(fig_roc, use_container_width=True)

st.divider()
st.caption("Smart Loan Risk Analyzer v2.0 | Managed by Rahul Kumar")