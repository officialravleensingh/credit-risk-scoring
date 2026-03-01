import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.preprocessing import load_data, preprocess_data, prepare_features, scale_features

def compare_models():
    print("Loading and preprocessing data...")
    df = load_data()
    df_processed, _ = preprocess_data(df)
    X, y = prepare_features(df_processed)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Accuracy': f'{accuracy*100:.2f}%',
            'Precision': f'{precision:.4f}',
            'Recall': f'{recall:.4f}',
            'F1-Score': f'{f1:.4f}',
            'ROC-AUC': f'{roc_auc:.4f}'
        })
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*90)
    print("MODEL COMPARISON RESULTS")
    print("="*90)
    print(results_df.to_string(index=False))
    print("="*90)

if __name__ == "__main__":
    compare_models()
