import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from utils.preprocessing import load_data, preprocess_data, prepare_features, scale_features

def train_model():
    print("Loading data...")
    df = load_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['loan_paid_back'].value_counts()}")
    
    print("\nPreprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    
    print("Preparing features...")
    X, y = prepare_features(df_processed)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Accuracy: {accuracy*100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Default', 'Paid Back']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nSaving model parameters...")
    with open('models/model_params.py', 'w') as f:
        f.write('import numpy as np\n\n')
        f.write('coef = np.array([\n')
        for val in model.coef_[0]:
            f.write(f'    {val},\n')
        f.write('])\n\n')
        f.write(f'intercept = {model.intercept_[0]}\n\n')
        f.write('scaler_mean = np.array([\n')
        for val in scaler.mean_:
            f.write(f'    {val},\n')
        f.write('])\n\n')
        f.write('scaler_scale = np.array([\n')
        for val in scaler.scale_:
            f.write(f'    {val},\n')
        f.write('])\n\n')
        f.write('label_encoders = {\n')
        for col, le in label_encoders.items():
            f.write(f'    "{col}": {{\n')
            for i, cls in enumerate(le.classes_):
                f.write(f'        "{cls}": {i},\n')
            f.write('    },\n')
        f.write('}\n')
    
    print("Training complete!")
    return accuracy, roc_auc

if __name__ == "__main__":
    train_model()
