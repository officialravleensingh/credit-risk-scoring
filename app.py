import streamlit as st
import numpy as np
import pandas as pd
from models.model_params import coef, intercept, scaler_mean, scaler_scale, label_encoders

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_credit_risk(input_data):
    input_df = pd.DataFrame([input_data])
    
    categorical_cols = ['gender', 'marital_status', 'education_level', 
                       'employment_status', 'loan_purpose', 'grade_subgrade']
    
    for col in categorical_cols:
        if col in label_encoders:
            input_df[col] = label_encoders[col][input_df[col].iloc[0]]
    
    input_scaled = (input_df.values - scaler_mean) / scaler_scale
    z = np.dot(input_scaled, coef) + intercept
    prob_paid = sigmoid(z)[0]
    prediction = 1 if prob_paid > 0.5 else 0
    probability = [1 - prob_paid, prob_paid]
    
    return prediction, probability

def main():
    st.title("💳 Credit Risk Scoring System")
    st.markdown("### Predict loan repayment probability using Machine Learning")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD", "Other"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed", "Student", "Retired"])
    
    with col2:
        st.subheader("💰 Financial Information")
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
        st.subheader("🏦 Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
        loan_purpose = st.selectbox("Loan Purpose", ["Debt consolidation", "Car", "Home", "Business", 
                                                      "Medical", "Education", "Vacation", "Other"])
        interest_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, 0.1)
        loan_term = st.selectbox("Loan Term (months)", [36, 60])
        installment = (loan_amount * (interest_rate/100/12) * (1 + interest_rate/100/12)**loan_term) / ((1 + interest_rate/100/12)**loan_term - 1)
        grade_subgrade = st.selectbox("Grade/Subgrade", ["A1", "A2", "A3", "A4", "A5",
                                                          "B1", "B2", "B3", "B4", "B5",
                                                          "C1", "C2", "C3", "C4", "C5",
                                                          "D1", "D2", "D3", "D4", "D5",
                                                          "E1", "E2", "E3", "E4", "E5",
                                                          "F1", "F2", "F3", "F4", "F5"])
    
    with col4:
        st.subheader("📊 Credit History")
        delinquency_history = st.number_input("Delinquency History", min_value=0, value=0)
        public_records = st.number_input("Public Records", min_value=0, value=0)
        num_of_delinquencies = st.number_input("Number of Delinquencies", min_value=0, value=0)
    
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button("🔍 Assess Credit Risk", use_container_width=True)
    
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
        
        if prediction == 1:
            st.success("✅ LOW RISK - Loan Likely to be Paid Back")
            st.balloons()
            st.info(f"Probability of repayment: {probability[1]*100:.2f}%")
        else:
            st.error("⚠️ HIGH RISK - Loan Default Likely")
            st.warning(f"Probability of default: {probability[0]*100:.2f}%")
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Training Samples", "20,000")
    with col_info2:
        st.metric("Algorithm", "Logistic Regression")
    with col_info3:
        st.metric("Features", "21")

if __name__ == "__main__":
    main()
    
    st.sidebar.title("ℹ️ About")
    st.sidebar.info(
        "This application predicts credit risk and loan repayment probability "
        "using machine learning trained on 20,000 historical loan applications."
    )
    st.sidebar.markdown("### 🎯 Features")
    st.sidebar.markdown(
        "- Real-time predictions\n"
        "- Comprehensive risk assessment\n"
        "- Multiple ML models\n"
        "- 21 input features"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Developed by:** Ravleen Singh & Team")
    st.sidebar.markdown("**GitHub:** [View Repository](https://github.com/officialravleensingh/credit-risk-scoring)")
    st.sidebar.markdown("**Project:** GenAI Capstone - Milestone 1")
