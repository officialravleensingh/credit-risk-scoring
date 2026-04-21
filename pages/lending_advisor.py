import streamlit as st
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env
load_dotenv()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.model_params import (
    label_encoders,
    n_estimators, max_depth, random_state
)

st.set_page_config(page_title="Lending Advisor", page_icon="", layout="wide")

FEATURE_ORDER = [
    'age', 'gender', 'marital_status', 'education_level', 'annual_income',
    'monthly_income', 'employment_status', 'debt_to_income_ratio', 'credit_score',
    'loan_amount', 'loan_purpose', 'interest_rate', 'loan_term', 'installment',
    'grade_subgrade', 'num_of_open_accounts', 'total_credit_limit', 'current_balance',
    'delinquency_history', 'public_records', 'num_of_delinquencies'
]
CATEGORICAL_COLS = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']


def _encode_categorical_value(column_name, value):
    mapping = label_encoders.get(column_name, {})
    if value in mapping:
        return mapping[value]
    if 'Other' in mapping:
        return mapping['Other']
    return next(iter(mapping.values()), 0)


@st.cache_resource
def load_model_and_scaler():
    from utils.preprocessing import load_data, preprocess_data, prepare_features
    df = load_data()
    df_processed, _ = preprocess_data(df)
    X, y = prepare_features(df_processed)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def get_ml_prediction(input_data: dict):
    input_df = pd.DataFrame([input_data])
    for col in CATEGORICAL_COLS:
        input_df[col] = _encode_categorical_value(col, input_df[col].iloc[0])
    input_array = input_df[FEATURE_ORDER].values.astype(float)
    model, scaler = load_model_and_scaler()
    input_scaled = scaler.transform(input_array)
    prediction = int(model.predict(input_scaled)[0])
    probability = model.predict_proba(input_scaled)[0]
    return prediction, float(probability[1]), float(probability[0])


#  Sidebar 
st.sidebar.title(" Lending Advisor")
st.sidebar.info(
    "This page uses an agentic AI workflow to generate a structured "
    "lending assessment report with regulatory references."
)
st.sidebar.markdown("###  Agent Workflow")
st.sidebar.markdown(
    "1. **Risk Analyzer** — Evaluates borrower profile\n"
    "2. **Regulation Retriever** — Fetches relevant guidelines (RAG)\n"
    "3. **Report Generator** — Produces structured assessment"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Framework:** LangGraph")
st.sidebar.markdown("**LLM:** Llama 3.3 70B (Groq)")
st.sidebar.markdown("**RAG:** FAISS + SentenceTransformers")
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** Ravleen Singh, Anurag Pandey, Ansh Tomar, Himanshu Chauhan")


#  Main UI 
st.title(" Agentic Lending Decision Support")
st.markdown("### AI-powered credit assessment with regulatory compliance")
st.markdown("---")

# Load Groq API key - try Streamlit secrets first (for Cloud), then .env (for local development)
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    groq_key = os.getenv('GROQ_API_KEY')

if not groq_key:
    st.error(
        "⚠️ **GROQ_API_KEY not configured**\n\n"
        "**For local development:**\n"
        "Create a `.env` file in project root:\n"
        "```\nGROQ_API_KEY=your_api_key_here\n```\n\n"
        "**For Streamlit Cloud:**\n"
        "Add secret in Streamlit Cloud settings:\n"
        "1. Go to app settings → Secrets\n"
        "2. Add: `GROQ_API_KEY=your_api_key_here`\n\n"
        "Get free API key: https://console.groq.com"
    )
    st.stop()

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=35, key="adv_age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="adv_gender")
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"], key="adv_ms")
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"], key="adv_edu")
    employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student", "Retired"], key="adv_emp")

with col2:
    st.subheader(" Financial Information")
    annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000, key="adv_inc")
    monthly_income = annual_income / 12
    debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.15, 0.01, key="adv_dti")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, key="adv_cs")
    num_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=5, key="adv_oa")
    total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0, value=50000, key="adv_tcl")
    current_balance = st.number_input("Current Balance ($)", min_value=0, value=10000, key="adv_cb")

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader(" Loan Details")
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000, key="adv_la")
    loan_purpose = st.selectbox("Loan Purpose", [
        "Debt consolidation", "Car", "Home", "Business",
        "Medical", "Education", "Vacation", "Other"
    ], key="adv_lp")
    interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, 0.1, key="adv_ir")
    loan_term = st.selectbox("Loan Term (months)", [36, 60], key="adv_lt")
    r = interest_rate / 100 / 12
    installment = (loan_amount * r * (1 + r) ** loan_term) / ((1 + r) ** loan_term - 1)
    grade_subgrade = st.selectbox("Grade/Subgrade", [
        "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5",
        "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5",
        "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5"
    ], key="adv_gs")

with col4:
    st.subheader(" Credit History")
    delinquency_history = st.number_input("Delinquency History", min_value=0, value=0, key="adv_dh")
    public_records = st.number_input("Public Records", min_value=0, value=0, key="adv_pr")
    num_of_delinquencies = st.number_input("Number of Delinquencies", min_value=0, value=0, key="adv_nd")

st.markdown("---")

_, col_btn, _ = st.columns([1, 1, 1])
with col_btn:
    run_button = st.button(" Generate Lending Assessment", use_container_width=True)

if run_button:
    os.environ['GROQ_API_KEY'] = groq_key

    input_data = {
        'age': age, 'gender': gender, 'marital_status': marital_status,
        'education_level': education_level, 'annual_income': annual_income,
        'monthly_income': monthly_income, 'employment_status': employment_status,
        'debt_to_income_ratio': debt_to_income_ratio, 'credit_score': credit_score,
        'loan_amount': loan_amount, 'loan_purpose': loan_purpose,
        'interest_rate': interest_rate, 'loan_term': loan_term,
        'installment': installment, 'grade_subgrade': grade_subgrade,
        'num_of_open_accounts': num_of_open_accounts, 'total_credit_limit': total_credit_limit,
        'current_balance': current_balance, 'delinquency_history': delinquency_history,
        'public_records': public_records, 'num_of_delinquencies': num_of_delinquencies
    }

    with st.spinner('Running ML model...'):
        prediction, repayment_prob, default_prob = get_ml_prediction(input_data)

    borrower_data = {**input_data,
                     'ml_prediction': prediction,
                     'repayment_probability': repayment_prob,
                     'default_probability': default_prob}

    st.markdown("---")
    st.markdown("###  ML Model Result")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        if prediction == 1:
            st.success(" LOW RISK")
        else:
            st.error(" HIGH RISK")
    with col_r2:
        st.metric("Repayment Probability", f"{repayment_prob*100:.1f}%")
    with col_r3:
        st.metric("Default Probability", f"{default_prob*100:.1f}%")

    st.markdown("---")
    st.markdown("###  Running Agentic Workflow...")

    progress = st.progress(0)
    status = st.empty()

    try:
        from agent.nodes import risk_analyzer_node, regulation_retriever_node, report_generator_node
        from agent.state import AgentState

        state = {'borrower': borrower_data, 'risk_summary': None,
                 'retrieved_regulations': None, 'final_report': None, 'messages': []}

        status.info(" Step 1/3 — Risk Analyzer: Evaluating borrower profile...")
        progress.progress(15)
        state.update(risk_analyzer_node(state))

        status.info(" Step 2/3 — Regulation Retriever: Fetching relevant guidelines...")
        progress.progress(50)
        state.update(regulation_retriever_node(state))

        status.info(" Step 3/3 — Report Generator: Producing structured assessment...")
        progress.progress(80)
        state.update(report_generator_node(state))

        result = {'risk_summary': state['risk_summary'],
                  'retrieved_regulations': state['retrieved_regulations'],
                  'final_report': state['final_report']}

        progress.progress(100)
        status.success(" Assessment complete!")

        st.markdown("---")
        st.markdown("##  Lending Assessment Report")

        report_sections = result['final_report'].split('##')
        for section in report_sections:
            section = section.strip()
            if not section:
                continue
            lines = section.split('\n', 1)
            title = lines[0].strip()
            content = lines[1].strip() if len(lines) > 1 else ''

            if 'BORROWER PROFILE' in title.upper():
                with st.expander(" Borrower Profile & Risk Analysis", expanded=True):
                    st.markdown(content)
            elif 'LENDING DECISION' in title.upper():
                with st.expander(" Lending Decision", expanded=True):
                    if 'APPROVE' in content.upper() or prediction == 1:
                        st.success(content)
                    else:
                        st.error(content)
            elif 'REGULATORY' in title.upper():
                with st.expander(" Regulatory References", expanded=True):
                    st.markdown(content)
            elif 'DISCLAIMER' in title.upper():
                with st.expander(" Legal Disclaimer", expanded=False):
                    st.warning(content)

        with st.expander(" View Retrieved Regulations (RAG Sources)", expanded=False):
            st.text(result['retrieved_regulations'])

    except Exception as e:
        progress.empty()
        status.error(f"Agent error: {str(e)}")
        st.exception(e)
