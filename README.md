# 💳 Intelligent Credit Risk Scoring System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automated credit risk prediction system using classical machine learning to assess loan repayment probability with **90.15% accuracy**.

🔗 **[Live Demo](https://credit-riskscoring.streamlit.app)** | 📂 **[GitHub Repository](https://github.com/officialravleensingh/credit-risk-scoring)**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Visualizations](#visualizations)
- [Team](#team)
- [Future Work](#future-work)

---

## 🎯 Project Overview

This project implements an intelligent credit risk assessment system that predicts the likelihood of loan repayment based on comprehensive borrower information. Traditional manual credit evaluation processes are time-consuming and prone to human bias. Our system automates this process using machine learning trained on **20,000 historical loan applications**, enabling faster and more reliable lending decisions.

### Key Achievements
- ✅ **90.15% Accuracy** with Random Forest model
- ✅ **0.8759 ROC-AUC Score** demonstrating excellent discrimination
- ✅ **Real-time predictions** through interactive web interface
- ✅ **Comprehensive model comparison** (Logistic Regression, Decision Tree, Random Forest)
- ✅ **Production-ready deployment** on Streamlit Cloud

---

## 🚨 Problem Statement

Financial institutions face significant challenges in evaluating loan applications:

| Challenge | Impact |
|-----------|--------|
| **Manual Assessment** | Slow, taking days or weeks per application |
| **Human Bias** | Inconsistent decisions across loan officers |
| **Scalability** | Difficulty processing large volumes efficiently |
| **Transparency** | Lack of clear decision criteria |

**Our Solution:** An automated, data-driven approach to credit risk assessment that delivers consistent, unbiased predictions in real-time.

---

## 📊 Dataset

### Overview

| Attribute | Value |
|-----------|-------|
| **Total Samples** | 20,000 loan applications |
| **Features** | 21 input features + 1 target variable |
| **Target Variable** | `loan_paid_back` (1 = paid, 0 = defaulted) |
| **Class Distribution** | 79.99% paid back, 20.01% defaulted |
| **Missing Values** | None (clean dataset) |
| **Data Quality** | High - no imputation required |

### Feature Categories

<table>
<tr>
<td width="50%">

**👤 Demographics**
- age
- gender
- marital_status
- education_level

**💰 Financial Information**
- annual_income
- monthly_income
- debt_to_income_ratio
- credit_score
- num_of_open_accounts
- total_credit_limit
- current_balance

</td>
<td width="50%">

**🏦 Loan Details**
- loan_amount
- loan_purpose
- interest_rate
- loan_term
- installment
- grade_subgrade

**📈 Credit History**
- employment_status
- delinquency_history
- public_records
- num_of_delinquencies

</td>
</tr>
</table>

### Key Insights from EDA

1. **Credit Score** - Strongest predictor of loan repayment
2. **Delinquency History** - Strong negative correlation with repayment
3. **Debt-to-Income Ratio** - Critical risk indicator
4. **Employment Status** - Employed borrowers show higher repayment rates
5. **Public Records** - Significant impact on default probability

---

## 🏆 Model Performance

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** ⭐ | **90.15%** | **0.8945** | **0.9941** | **0.9417** | **0.8759** |
| Logistic Regression | 88.78% | 0.8995 | 0.9678 | 0.9324 | 0.8515 |
| Decision Tree | 88.52% | 0.8983 | 0.9659 | 0.9309 | 0.8406 |

⭐ **Best Model: Random Forest** - Selected for deployment

### Detailed Performance (Random Forest)

| Metric | Default (0) | Paid Back (1) |
|--------|-------------|---------------|
| **Precision** | 0.96 | 0.89 |
| **Recall** | 0.53 | 0.99 |
| **F1-Score** | 0.68 | 0.94 |
| **Support** | 800 | 3,200 |

### Confusion Matrix

```
                Predicted
              Default  Paid Back
Actual Default    425       375
    Paid Back      19      3181
```

**Interpretation:**
- **True Positives (Paid Back):** 3,181 (99% recall)
- **True Negatives (Default):** 425 (53% recall)
- **False Positives:** 375 (conservative - better for risk management)
- **False Negatives:** 19 (minimal - only 0.6% of paid-back loans misclassified)

---

## ✨ Features

### Core Capabilities
- 🎯 **Automated Risk Assessment** - Instant credit risk evaluation
- 📊 **Multi-Model Comparison** - Tested 3 ML algorithms
- 🔄 **Real-time Predictions** - Sub-second response time
- 📈 **Visual Analytics** - Confusion matrices, ROC curves, feature importance
- 🌐 **Cloud Deployment** - Accessible worldwide 24/7
- 🔒 **Bias-Free Decisions** - Consistent evaluation criteria

### Technical Features
- Custom data preprocessing pipeline
- Label encoding for 6 categorical features
- Standard scaling for numerical features
- Stratified train-test split (80-20)
- Feature importance analysis
- Model performance visualization

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.13 |
| **ML Framework** | scikit-learn 1.3.0 |
| **Data Processing** | pandas 2.1.0, numpy 1.26.0 |
| **Visualization** | matplotlib 3.7.0, seaborn 0.12.0 |
| **Web Framework** | Streamlit 1.28.0 |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git, GitHub |
| **Model Storage** | Python modules (.py format) |

---

## 📥 Installation

### Prerequisites
- Python 3.13 or higher
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/officialravleensingh/credit-risk-scoring.git
cd credit-risk-scoring
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python --version  # Should show Python 3.13+
```

---

## 🚀 Usage

### 1. Train the Model

```bash
python train_model.py
```

**Output:**
- Trains Random Forest model
- Generates performance metrics
- Creates visualizations in `visualizations/` folder
- Saves model parameters to `models/model_params.py`

### 2. Compare Models

```bash
python compare_models.py
```

**Output:**
- Compares Logistic Regression, Decision Tree, and Random Forest
- Generates comparison visualizations
- Displays performance table

### 3. Run Web Application

```bash
streamlit run app.py
```

**Access:** Open browser at `http://localhost:8501`

### 4. Using the Application

**Step 1:** Enter borrower information
- Personal Information (age, gender, education, etc.)
- Financial Information (income, credit score, etc.)
- Loan Details (amount, purpose, interest rate, etc.)
- Credit History (delinquencies, public records, etc.)

**Step 2:** Click "Assess Credit Risk"

**Step 3:** View results
- Risk classification (Low Risk / High Risk)
- Probability scores
- Visual indicators

---

## 📁 Project Structure

```
credit-risk-scoring/
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script (Random Forest)
├── compare_models.py               # Model comparison script
├── utils/
│   └── preprocessing.py            # Data preprocessing functions
├── models/
│   └── model_params.py             # Trained model parameters
├── dataset/
│   └── original_dataset.csv        # Training data (20,000 samples)
├── visualizations/                 # Generated visualizations
│   ├── confusion_matrices.png      # All models confusion matrices
│   ├── roc_curves.png              # ROC curves comparison
│   ├── metrics_comparison.png      # Performance metrics chart
│   ├── feature_importance.png      # Feature importance chart
│   ├── final_confusion_matrix.png  # Final model confusion matrix
│   ├── final_roc_curve.png         # Final model ROC curve
│   └── final_feature_importance.png # Final model feature importance
├── notebooks/
│   └── eda.ipynb                   # Exploratory data analysis
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Categorical Encoding:**
- Applied Label Encoding to 6 categorical features
- Converted text categories to numerical values

**Feature Scaling:**
- Used StandardScaler for all numerical features
- Formula: `z = (x - μ) / σ`
- Ensures equal contribution from all features

**Train-Test Split:**
- 80% training (16,000 samples)
- 20% testing (4,000 samples)
- Stratified split to maintain class distribution

### 2. Model Selection & Training

**Models Evaluated:**
1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Non-linear decision boundaries
3. **Random Forest** - Ensemble of decision trees (WINNER)

**Hyperparameters (Random Forest):**
- `n_estimators`: 100 trees
- `max_depth`: 10 levels
- `random_state`: 42 (reproducibility)
- `n_jobs`: -1 (parallel processing)

### 3. Model Evaluation

**Metrics Used:**
- Accuracy - Overall correctness
- Precision - Positive prediction reliability
- Recall - True positive detection rate
- F1-Score - Harmonic mean of precision and recall
- ROC-AUC - Discrimination capability

**Validation Strategy:**
- Stratified K-Fold cross-validation
- Confusion matrix analysis
- ROC curve visualization

### 4. Deployment

**Architecture:**
- Model parameters stored as Python module
- Streamlit web interface for user interaction
- Cloud deployment on Streamlit Cloud
- Real-time prediction engine

---

## 📈 Visualizations

### Generated Visualizations

1. **Confusion Matrices** - Visual representation of predictions vs actual
2. **ROC Curves** - Model performance across thresholds
3. **Metrics Comparison** - Bar chart comparing all models
4. **Feature Importance** - Top 10 most influential features

All visualizations are automatically generated during training and displayed in the Streamlit app sidebar.

---

## 👥 Team

| Name | Role | Contributions |
|------|------|---------------|
| **Ravleen Singh** | Project Lead | Model development, training pipeline, deployment, integration |
| **Anurag Pandey** | Data Engineer | Data preprocessing, feature engineering, categorical encoding |
| **Ansh Tomar** | Data Analyst | EDA, visualization, documentation, insights generation |
| **Himanshu Chauhan** | Frontend Developer | UI development, Streamlit app design, testing, UX optimization |

---

## 🔮 Future Work

### Planned Enhancements

1. **Advanced Models**
   - Implement XGBoost and LightGBM
   - Neural network architectures
   - Ensemble stacking methods

2. **Explainability**
   - SHAP values for individual predictions
   - LIME for local interpretability
   - Feature contribution visualization

3. **Features**
   - Time-series analysis of borrower behavior
   - Alternative credit scoring (social media, utility bills)
   - Batch prediction capability

4. **Integration**
   - REST API development
   - Banking system integration
   - Real-time monitoring dashboard

5. **Optimization**
   - Model compression for faster inference
   - A/B testing framework
   - Automated retraining pipeline

---

## 📄 License

This project is part of the **GenAI Capstone Project (Milestone 1)** at Newton School of Technology.

---

## 🙏 Acknowledgments

- **Newton School of Technology** - Project guidance and support
- **Streamlit** - Deployment platform
- **scikit-learn** - Machine learning framework
- **Dataset Source** - Synthetic credit risk data

---

## 📞 Contact

- **GitHub:** [officialravleensingh](https://github.com/officialravleensingh)
- **Live Demo:** [credit-riskscoring.streamlit.app](https://credit-riskscoring.streamlit.app)
- **Project Type:** Classical Machine Learning (No GenAI)
- **Institution:** Newton School of Technology
- **Date:** February 2025

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by Team Credit Risk Scoring

</div>
