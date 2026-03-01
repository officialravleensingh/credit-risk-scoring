import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath='dataset/original_dataset.csv'):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df = df.copy()
    
    categorical_cols = ['gender', 'marital_status', 'education_level', 
                       'employment_status', 'loan_purpose', 'grade_subgrade']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders

def prepare_features(df, target_col='loan_paid_back'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
