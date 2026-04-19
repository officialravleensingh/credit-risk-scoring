import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from models.model_params import (
    feature_importances, label_encoders, accuracy, roc_auc,
    n_estimators, max_depth, random_state
)
from PIL import Image

st.set_page_config(page_title="Credit Risk Scoring", page_icon="", layout="wide")

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = load_data()
    df_processed, _ = preprocess_data(df)
    X, y = prepare_features(df_processed)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler

def predict_credit_risk(input_data):
    input_df = pd.DataFrame([input_data])

    for col in CATEGORICAL_COLS:
        input_df[col] = _encode_categorical_value(col, input_df[col].iloc[0])

    input_array = input_df[FEATURE_ORDER].values.astype(float)

    model, scaler = load_model_and_scaler()
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability


#  Sidebar 
st.sidebar.title("ℹ About")
st.sidebar.info(
    "This application predicts credit risk and loan repayment probability "
    "using Random Forest machine learning trained on 20,000 historical loan applications."
)

st.sidebar.markdown("###  Model Performance")
try:
    img_cm = Image.open('visualizations/final_confusion_matrix.png')
    st.sidebar.image(img_cm, caption='Confusion Matrix', use_container_width=True)
    img_roc = Image.open('visualizations/final_roc_curve.png')
    st.sidebar.image(img_roc, caption='ROC Curve', use_container_width=True)
except Exception:
    st.sidebar.warning("Visualizations not found. Run train_model.py first.")

st.sidebar.markdown("###  Features")
st.sidebar.markdown(
    "- Real-time predictions\n"
    "- 90.15% accuracy\n"
    "- Random Forest algorithm\n"
    "- 21 input features"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** Ravleen Singh, Anurag Pandey, Ansh Tomar, Himanshu Chauhan")
st.sidebar.markdown("**GitHub:** [View Repository](https://github.com/officialravleensingh/credit-risk-scoring)")
st.sidebar.markdown("**Project:** GenAI Capstone - Milestone 1")


#  Main UI 
def main():
    st.title(" Credit Risk Scoring System")
    st.markdown("### Predict loan repayment probability using Machine Learning")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student", "Retired"])

    with col2:
        st.subheader(" Financial Information")
        annual_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        monthly_income = annual_income / 12
        debt_to_income_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.15, 0.01)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        num_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=5)
        total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0, value=50000)
        current_balance = st.number_input("Current Balance ($)", min_value=0, value=10000)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader(" Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
        loan_purpose = st.selectbox("Loan Purpose", [
            "Debt consolidation", "Car", "Home", "Business",
            "Medical", "Education", "Vacation", "Other"
        ])
        interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, 0.1)
        loan_term = st.selectbox("Loan Term (months)", [36, 60])
        r = interest_rate / 100 / 12
        installment = (loan_amount * r * (1 + r) ** loan_term) / ((1 + r) ** loan_term - 1)
        grade_subgrade = st.selectbox("Grade/Subgrade", [
            "A1", "A2", "A3", "A4", "A5",
            "B1", "B2", "B3", "B4", "B5",
            "C1", "C2", "C3", "C4", "C5",
            "D1", "D2", "D3", "D4", "D5",
            "E1", "E2", "E3", "E4", "E5",
            "F1", "F2", "F3", "F4", "F5"
        ])

    with col4:
        st.subheader(" Credit History")
        delinquency_history = st.number_input("Delinquency History", min_value=0, value=0)
        public_records = st.number_input("Public Records", min_value=0, value=0)
        num_of_delinquencies = st.number_input("Number of Delinquencies", min_value=0, value=0)

    st.markdown("---")

    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        predict_button = st.button(" Assess Credit Risk", use_container_width=True)

    if predict_button:
        input_data = {
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'education_level': education_level,
            'annual_income': annual_income,
            'monthly_income': monthly_income,
            'employment_status': employment_status,
            'debt_to_income_ratio': debt_to_income_ratio,
            'credit_score': credit_score,
            'loan_amount': loan_amount,
            'loan_purpose': loan_purpose,
            'interest_rate': interest_rate,
            'loan_term': loan_term,
            'installment': installment,
            'grade_subgrade': grade_subgrade,
            'num_of_open_accounts': num_of_open_accounts,
            'total_credit_limit': total_credit_limit,
            'current_balance': current_balance,
            'delinquency_history': delinquency_history,
            'public_records': public_records,
            'num_of_delinquencies': num_of_delinquencies
        }

        with st.spinner('Analyzing credit risk...'):
            prediction, probability = predict_credit_risk(input_data)

        st.markdown("---")

        col_result1, col_result2 = st.columns(2)
        with col_result1:
            if prediction == 1:
                st.success(" LOW RISK - Loan Likely to be Paid Back")
                st.balloons()
            else:
                st.error(" HIGH RISK - Loan Default Likely")

        with col_result2:
            st.metric("Repayment Probability", f"{probability[1]*100:.2f}%")
            st.metric("Default Probability", f"{probability[0]*100:.2f}%")

    st.markdown("---")
    st.markdown("###  Model Information")
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Training Samples", "20,000")
    with col_info2:
        st.metric("Algorithm", "Random Forest")
    with col_info3:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
    with col_info4:
        st.metric("ROC-AUC", f"{roc_auc:.4f}")


main()
