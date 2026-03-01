# Intelligent Credit Risk Scoring System

Automated credit risk prediction system using classical machine learning to assess loan repayment probability.

## Project Overview

This project implements an intelligent credit risk assessment system that predicts the likelihood of loan repayment based on comprehensive borrower information. Traditional manual credit evaluation processes are time-consuming and prone to human bias. Our system automates this process using machine learning trained on 20,000 historical loan applications, enabling faster and more reliable lending decisions.

## Problem Statement

Financial institutions face significant challenges in evaluating loan applications:
- Manual assessment processes are slow and resource-intensive
- Human bias can affect lending decisions
- Inconsistent evaluation criteria across different loan officers
- Difficulty in processing large volumes of applications efficiently

Our solution provides an automated, data-driven approach to credit risk assessment that delivers consistent, unbiased predictions in real-time.

## Features

- Automated credit risk assessment
- Real-time loan repayment probability prediction
- Interactive web interface for easy data input
- Comprehensive data preprocessing pipeline
- Model trained on 20,000 historical loan records
- 21 feature analysis including demographics, financials, and credit history

## Tech Stack

- **Python 3.13**
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Model Storage**: Python modules (.py format)

## Dataset

- **Size**: 20,000 loan applications
- **Features**: 21 input features + 1 target variable
- **Target**: loan_paid_back (1 = paid back, 0 = defaulted)
- **Class Distribution**: 79.99% paid back, 20.01% defaulted
- **Data Quality**: No missing values, clean dataset

### Feature Categories:

**Demographics:**
- age, gender, marital_status, education_level

**Financial Information:**
- annual_income, monthly_income, debt_to_income_ratio, credit_score
- num_of_open_accounts, total_credit_limit, current_balance

**Loan Details:**
- loan_amount, loan_purpose, interest_rate, loan_term, installment, grade_subgrade

**Credit History:**
- employment_status, delinquency_history, public_records, num_of_delinquencies

## Project Structure

```
credit-risk-scoring/
├── app.py                    # Streamlit web application
├── train_model.py            # Model training script
├── utils/
│   └── preprocessing.py      # Data preprocessing functions
├── models/
│   └── model_params.py       # Trained model parameters
├── dataset/
│   └── original_dataset.csv  # Training data (20,000 samples)
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/officialravleensingh/credit-risk-scoring.git
cd credit-risk-scoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train Logistic Regression model
- Evaluate model performance
- Save model parameters to `models/model_params.py`

### Running the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. Enter borrower information across four categories:
   - Personal Information (age, gender, education, etc.)
   - Financial Information (income, credit score, etc.)
   - Loan Details (amount, purpose, interest rate, etc.)
   - Credit History (delinquencies, public records, etc.)

2. Click "Assess Credit Risk" button

3. View prediction results:
   - Risk classification (Low Risk / High Risk)
   - Probability of loan repayment or default

## Model Performance

### Best Model: Logistic Regression

- **Accuracy**: 88.78%
- **ROC-AUC Score**: 0.8515
- **Precision (Paid Back)**: 0.90
- **Recall (Paid Back)**: 0.97
- **F1-Score (Paid Back)**: 0.93

### Confusion Matrix:
```
                Predicted
              Default  Paid Back
Actual Default    454       346
    Paid Back     103      3097
```

### Key Findings from EDA:

1. **Credit Score**: Strongest predictor of loan repayment
2. **Delinquency History**: Strong negative correlation with repayment
3. **Debt-to-Income Ratio**: Important risk indicator
4. **Employment Status**: Employed borrowers show higher repayment rates
5. **Public Records**: Significant impact on default probability

## Methodology

### 1. Data Preprocessing
- Label encoding for categorical features (gender, marital_status, education_level, employment_status, loan_purpose, grade_subgrade)
- Standard scaling for numerical features
- Train-test split (80-20) with stratification

### 2. Model Training
- Algorithm: Logistic Regression
- Max iterations: 1000
- Random state: 42 for reproducibility

### 3. Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Cross-validation for generalization assessment

### 4. Deployment
- Model parameters stored as Python module
- Streamlit web interface for user interaction
- Cloud deployment on Streamlit Cloud

## Deployment

**Live Application**: [https://credit-riskscoring.streamlit.app](https://credit-riskscoring.streamlit.app)

The application is deployed on Streamlit Cloud and accessible worldwide for real-time credit risk assessment.

## Team Contributors

- **Ravleen Singh** - Project Lead, Model Development, Deployment
- **Anurag Pandey** - Data Preprocessing, Feature Engineering
- **Ansh Tomar** - EDA, Visualization, Documentation
- **Himanshu Chauhan** - UI Development, Testing, Integration

## Project Timeline

- **Week 1**: Dataset sourcing, EDA, preprocessing
- **Week 2**: Model training, evaluation, optimization
- **Week 3**: Web application development, deployment
- **Week 4**: Documentation, testing, final presentation

## Future Enhancements

- Implement ensemble models (Random Forest, XGBoost)
- Add SHAP values for model interpretability
- Include time-series analysis for temporal patterns
- Develop API for integration with banking systems
- Add batch prediction capability for multiple applications

## License

This project is part of the GenAI Capstone Project (Milestone 1) at Newton School of Technology.

## Acknowledgments

- Newton School of Technology for project guidance
- Dataset source: Synthetic credit risk data
- Streamlit for deployment platform

---

**Project Type**: Classical Machine Learning (No GenAI)  
**Course**: GenAI Capstone Project  
**Institution**: Newton School of Technology  
**Date**: February 2025
