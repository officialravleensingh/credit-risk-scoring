# Intelligent Credit Risk Scoring System

ML-based credit risk prediction system using classical machine learning algorithms.

## Project Overview

This project implements an automated credit risk assessment system that predicts loan repayment probability based on borrower information. The system uses classical machine learning techniques to analyze historical loan data and assess credit risk.

## Features

- Data preprocessing and feature engineering
- Multiple ML models (Logistic Regression, Random Forest, Decision Tree)
- Interactive web interface using Streamlit
- Real-time credit risk predictions
- Model comparison and evaluation

## Tech Stack

- **Python 3.13**
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: joblib

## Dataset

- **Rows**: 20,000 loan applications
- **Features**: 21 input features + 1 target
- **Target**: loan_paid_back (1 = paid, 0 = defaulted)
- **Class Distribution**: 79.99% paid back, 20.01% defaulted

### Key Features:
- Demographics: age, gender, marital_status, education_level
- Financial: annual_income, monthly_income, debt_to_income_ratio, credit_score
- Loan Details: loan_amount, loan_purpose, interest_rate, loan_term
- Credit History: delinquency_history, public_records, num_of_delinquencies

## Project Structure

```
credit-risk-scoring/
├── app.py                    # Streamlit web application
├── train_model.py            # Model training script
├── compare_models.py         # Model comparison
├── utils/
│   └── preprocessing.py      # Data preprocessing functions
├── models/
│   ├── model.joblib          # Trained model
│   ├── scaler.joblib         # Fitted scaler
│   └── label_encoders.joblib # Label encoders
├── dataset/
│   └── original_dataset.csv  # Training data
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

### Running the Web App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Model Performance

- **Best Model**: Logistic Regression
- **Accuracy**: 88.78%
- **ROC-AUC**: 0.8515
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Deployment

This application is deployed on Streamlit Cloud: [Live Demo](https://credit-riskscoring.streamlit.app)

## Team

- Ravleen Singh
- Anurag Pandey
- Ansh Tomar
- Himanshu Chauhan

## License

This project is part of the GenAI Capstone Project (Milestone 1) at Newton School of Technology.
