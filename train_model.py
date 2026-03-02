import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from utils.preprocessing import load_data, preprocess_data, prepare_features, scale_features

def plot_confusion_matrix(y_test, y_pred, accuracy):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualizations/final_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/final_confusion_matrix.png")

def plot_roc_curve(y_test, y_pred_proba, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/final_roc_curve.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/final_roc_curve.png")

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/final_feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/final_feature_importance.png")

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
    
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
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
    
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, y_pred, accuracy)
    plot_roc_curve(y_test, y_pred_proba, roc_auc)
    plot_feature_importance(model, X.columns.tolist())
    
    print("\nSaving model parameters...")
    with open('models/model_params.py', 'w') as f:
        f.write('import numpy as np\n\n')
        f.write(f'model_type = "RandomForest"\n')
        f.write(f'n_estimators = {model.n_estimators}\n')
        f.write(f'max_depth = {model.max_depth}\n')
        f.write(f'random_state = {model.random_state}\n\n')
        f.write('feature_importances = np.array([\n')
        for val in model.feature_importances_:
            f.write(f'    {val},\n')
        f.write('])\n\n')
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
        f.write('}\n\n')
        f.write(f'accuracy = {accuracy}\n')
        f.write(f'roc_auc = {roc_auc}\n')
    
    print("Training complete!")
    return accuracy, roc_auc

if __name__ == "__main__":
    train_model()
