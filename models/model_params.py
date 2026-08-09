model_type = "RandomForest"
n_estimators = 100
max_depth = 10
random_state = 42

feature_names = ['employment_status', 'debt_to_income_ratio', 'credit_score', 'interest_rate', 'grade_subgrade', 'delinquency_history', 'annual_income', 'num_of_delinquencies', 'monthly_income', 'installment', 'loan_amount', 'total_credit_limit', 'public_records', 'num_of_open_accounts', 'current_balance', 'age', 'loan_term', 'gender', 'education_level', 'marital_status', 'loan_purpose']
feature_importances = {
    "employment_status": 0.6967495159363668,
    "debt_to_income_ratio": 0.17843470225450736,
    "credit_score": 0.08872779549259206,
    "interest_rate": 0.010379288099888634,
    "grade_subgrade": 0.007423057090099924,
    "delinquency_history": 0.004907130882728162,
    "annual_income": 0.0032566965187336904,
    "num_of_delinquencies": 0.002671355692344784,
    "monthly_income": 0.002286325205252778,
    "installment": 0.002120029829232831,
    "loan_amount": 0.0018134227296958615,
    "total_credit_limit": 0.001166902212413394,
    "public_records": 6.377805614375956e-05,
    "num_of_open_accounts": 0.0,
    "current_balance": 0.0,
    "age": 0.0,
    "loan_term": 0.0,
    "gender": 0.0,
    "education_level": 0.0,
    "marital_status": 0.0,
    "loan_purpose": 0.0,
}

top_features = [('employment_status', 0.6967495159363668), ('debt_to_income_ratio', 0.17843470225450736), ('credit_score', 0.08872779549259206), ('interest_rate', 0.010379288099888634), ('grade_subgrade', 0.007423057090099924), ('delinquency_history', 0.004907130882728162), ('annual_income', 0.0032566965187336904), ('num_of_delinquencies', 0.002671355692344784), ('monthly_income', 0.002286325205252778), ('installment', 0.002120029829232831)]
accuracy = 0.89775
roc_auc = 0.8737933593749999
