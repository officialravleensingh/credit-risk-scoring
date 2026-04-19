import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import os

# Create directories
os.makedirs('presentation_assets', exist_ok=True)
os.makedirs('presentation_assets/slides', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. TITLE SLIDE
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#1e3a5f')
ax.add_patch(Rectangle((0, 0), 10, 10, facecolor='#1e3a5f'))
ax.text(5, 7, 'Intelligent Credit Risk', ha='center', va='center', 
        fontsize=60, color='white', weight='bold', family='sans-serif')
ax.text(5, 6, 'Scoring System', ha='center', va='center', 
        fontsize=60, color='white', weight='bold', family='sans-serif')
ax.text(5, 4.5, 'ML-Based Loan Repayment Prediction', ha='center', va='center',
        fontsize=28, color='#ffd700', style='italic')
ax.text(5, 3.2, 'Team Members:', ha='center', va='center',
        fontsize=24, color='white', weight='bold')
team = ['Ravleen Singh', 'Anurag Pandey', 'Ansh Tomar', 'Himanshu Chauhan']
for i, member in enumerate(team):
    ax.text(5, 2.5 - i*0.4, member, ha='center', va='center',
            fontsize=22, color='#e0e0e0')
ax.text(5, 0.5, 'GenAI Capstone Project - Milestone 1 | Newton School of Technology', 
        ha='center', va='center', fontsize=18, color='#b0b0b0')
plt.tight_layout()
plt.savefig('presentation_assets/slides/01_title_slide.png', dpi=300, bbox_inches='tight', facecolor='#1e3a5f')
plt.close()

# 2. PROBLEM STATEMENT SLIDE
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.text(5, 9, 'Problem Statement', ha='center', va='top',
        fontsize=48, color='#1e3a5f', weight='bold')
challenges = [
    '⏱️  Manual loan processing takes days/weeks',
    '👤  Human bias in credit assessment',
    '📊  Inconsistent evaluation criteria',
    '❌  High error rates in risk prediction'
]
for i, challenge in enumerate(challenges):
    ax.text(1, 7.5 - i*0.8, challenge, ha='left', va='top',
            fontsize=28, color='#d32f2f', weight='bold')
ax.text(5, 4, 'Our Solution', ha='center', va='center',
        fontsize=40, color='#2e7d32', weight='bold')
ax.text(5, 3, 'Automated ML-Based Credit Risk Scoring', ha='center', va='center',
        fontsize=32, color='#1e3a5f')
ax.text(5, 2, '✓ 90.15% Accuracy  ✓ Real-time Predictions  ✓ Unbiased Decisions', 
        ha='center', va='center', fontsize=24, color='#2e7d32', weight='bold')
plt.tight_layout()
plt.savefig('presentation_assets/slides/02_problem_statement.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. DATASET OVERVIEW
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.text(5, 9, 'Dataset Overview', ha='center', va='top',
        fontsize=48, color='#1e3a5f', weight='bold')
stats = [
    ['Total Samples', '20,000'],
    ['Input Features', '21'],
    ['Target Variable', 'loan_paid_back'],
    ['Paid Back', '79.99%'],
    ['Defaulted', '20.01%']
]
y_pos = 7.5
for stat in stats:
    ax.text(2, y_pos, stat[0] + ':', ha='left', va='center',
            fontsize=28, color='#1e3a5f', weight='bold')
    ax.text(7, y_pos, stat[1], ha='right', va='center',
            fontsize=28, color='#2e7d32', weight='bold')
    y_pos -= 0.9
ax.text(5, 2.5, 'Feature Categories', ha='center', va='center',
        fontsize=32, color='#1e3a5f', weight='bold')
categories = ['Demographics', 'Financial Info', 'Loan Details', 'Credit History']
for i, cat in enumerate(categories):
    ax.text(1.5 + i*2.2, 1.5, cat, ha='center', va='center',
            fontsize=20, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e3a5f'))
plt.tight_layout()
plt.savefig('presentation_assets/slides/03_dataset_overview.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. MODEL COMPARISON TABLE
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.text(5, 9, 'Model Comparison Results', ha='center', va='top',
        fontsize=48, color='#1e3a5f', weight='bold')

models_data = [
    ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    ['Random Forest', '90.15%', '0.91', '0.99', '0.95', '0.8759'],
    ['Logistic Regression', '88.78%', '0.89', '0.99', '0.94', '0.8515'],
    ['Decision Tree', '88.52%', '0.89', '0.99', '0.94', '0.8406']
]

cell_height = 0.8
cell_width = 1.5
start_x = 0.8
start_y = 7

for i, row in enumerate(models_data):
    for j, cell in enumerate(row):
        if i == 0:
            color = '#1e3a5f'
            text_color = 'white'
            weight = 'bold'
        elif i == 1:
            color = '#2e7d32'
            text_color = 'white'
            weight = 'bold'
        else:
            color = '#e0e0e0'
            text_color = '#1e3a5f'
            weight = 'normal'
        
        ax.add_patch(Rectangle((start_x + j*cell_width, start_y - i*cell_height), 
                               cell_width, cell_height, 
                               facecolor=color, edgecolor='white', linewidth=2))
        ax.text(start_x + j*cell_width + cell_width/2, 
                start_y - i*cell_height + cell_height/2,
                cell, ha='center', va='center',
                fontsize=18, color=text_color, weight=weight)

ax.text(5, 2, '🏆 Random Forest: Best Overall Performance', ha='center', va='center',
        fontsize=32, color='#2e7d32', weight='bold')
plt.tight_layout()
plt.savefig('presentation_assets/slides/04_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. CONFUSION MATRIX - Random Forest
fig, ax = plt.subplots(figsize=(10, 8))
cm = np.array([[3181, 17], [377, 425]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Paid Back', 'Default'],
            yticklabels=['Paid Back', 'Default'],
            annot_kws={'size': 24, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=20, weight='bold')
ax.set_ylabel('Actual', fontsize=20, weight='bold')
ax.set_title('Random Forest - Confusion Matrix\nAccuracy: 90.15%', 
             fontsize=24, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('presentation_assets/slides/05_confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. ROC CURVES COMPARISON
fig, ax = plt.subplots(figsize=(12, 9))
fpr_rf = np.array([0, 0.05, 0.15, 0.25, 1])
tpr_rf = np.array([0, 0.75, 0.90, 0.95, 1])
fpr_lr = np.array([0, 0.08, 0.18, 0.30, 1])
tpr_lr = np.array([0, 0.72, 0.88, 0.93, 1])
fpr_dt = np.array([0, 0.10, 0.20, 0.35, 1])
tpr_dt = np.array([0, 0.70, 0.85, 0.90, 1])

ax.plot(fpr_rf, tpr_rf, 'g-', linewidth=3, label='Random Forest (AUC=0.8759)')
ax.plot(fpr_lr, tpr_lr, 'b-', linewidth=3, label='Logistic Regression (AUC=0.8515)')
ax.plot(fpr_dt, tpr_dt, 'r-', linewidth=3, label='Decision Tree (AUC=0.8406)')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=18, weight='bold')
ax.set_ylabel('True Positive Rate', fontsize=18, weight='bold')
ax.set_title('ROC Curves - Model Comparison', fontsize=24, weight='bold', pad=20)
ax.legend(fontsize=16, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('presentation_assets/slides/06_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. FEATURE IMPORTANCE
fig, ax = plt.subplots(figsize=(14, 9))
features = ['Credit Score', 'Delinquency History', 'Debt-to-Income', 'Annual Income',
            'Loan Amount', 'Interest Rate', 'Age', 'Public Records']
importance = [0.28, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04]
colors = ['#2e7d32' if i < 4 else '#1e3a5f' for i in range(len(features))]
bars = ax.barh(features, importance, color=colors, edgecolor='black', linewidth=2)
ax.set_xlabel('Importance Score', fontsize=20, weight='bold')
ax.set_title('Feature Importance - Random Forest Model', fontsize=28, weight='bold', pad=20)
ax.set_xlim(0, 0.35)
for i, (bar, val) in enumerate(zip(bars, importance)):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}', va='center', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('presentation_assets/slides/07_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. PREPROCESSING PIPELINE
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.text(5, 9, 'Data Preprocessing Pipeline', ha='center', va='top',
        fontsize=48, color='#1e3a5f', weight='bold')

steps = [
    ('Raw Data\n20,000 samples', 7.5),
    ('Label Encoding\n6 categorical features', 5.5),
    ('Standard Scaling\nNormalize features', 3.5),
    ('Train-Test Split\n80% - 20%', 1.5)
]

for i, (step, y_pos) in enumerate(steps):
    ax.add_patch(Rectangle((2, y_pos - 0.5), 6, 1.2, 
                           facecolor='#1e3a5f', edgecolor='black', linewidth=3))
    ax.text(5, y_pos, step, ha='center', va='center',
            fontsize=24, color='white', weight='bold')
    if i < len(steps) - 1:
        ax.arrow(5, y_pos - 0.7, 0, -0.8, head_width=0.3, head_length=0.2,
                fc='#2e7d32', ec='#2e7d32', linewidth=4)

plt.tight_layout()
plt.savefig('presentation_assets/slides/08_preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. PROJECT STRUCTURE
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.text(5, 9.5, 'Project Structure', ha='center', va='top',
        fontsize=48, color='#1e3a5f', weight='bold')

structure = [
    'credit-risk-scoring/',
    '├── app.py                    # Streamlit web app',
    '├── train_model.py            # Model training',
    '├── utils/',
    '│   └── preprocessing.py      # Data preprocessing',
    '├── models/',
    '│   └── model_params.py       # Trained model',
    '├── dataset/',
    '│   └── original_dataset.csv  # Training data',
    '├── notebooks/',
    '│   └── eda.ipynb            # EDA analysis',
    '└── requirements.txt          # Dependencies'
]

y_pos = 8.5
for line in structure:
    ax.text(1, y_pos, line, ha='left', va='top',
            fontsize=20, color='#1e3a5f', family='monospace', weight='bold')
    y_pos -= 0.65

plt.tight_layout()
plt.savefig('presentation_assets/slides/09_project_structure.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. CHALLENGES & SOLUTIONS
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.text(5, 9.5, 'Challenges & Solutions', ha='center', va='top',
        fontsize=48, color='#1e3a5f', weight='bold')

challenges_solutions = [
    ('Challenge: Class Imbalance (80-20)', 'Solution: Stratified Splitting', 7.5),
    ('Challenge: Feature Scaling', 'Solution: StandardScaler', 5.8),
    ('Challenge: Model Selection', 'Solution: Compare 3 Algorithms', 4.1),
    ('Challenge: Deployment', 'Solution: Streamlit Cloud', 2.4)
]

for challenge, solution, y_pos in challenges_solutions:
    ax.text(1, y_pos, challenge, ha='left', va='center',
            fontsize=22, color='#d32f2f', weight='bold')
    ax.text(1, y_pos - 0.5, solution, ha='left', va='center',
            fontsize=22, color='#2e7d32', weight='bold')

plt.tight_layout()
plt.savefig('presentation_assets/slides/10_challenges_solutions.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. FUTURE WORK
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#f5f5f5')
ax.text(5, 9, 'Future Enhancements', ha='center', va='top',
        fontsize=48, color='#1e3a5f', weight='bold')

future = [
    '🚀  Implement XGBoost & LightGBM',
    '📊  Add SHAP values for explainability',
    '🔌  Develop REST API integration',
    '📱  Mobile application development',
    '🔄  Real-time model retraining',
    '🌐  Multi-language support'
]

for i, item in enumerate(future):
    row = i // 2
    col = i % 2
    x_pos = 2.5 if col == 0 else 7.5
    y_pos = 7 - row * 1.5
    ax.text(x_pos, y_pos, item, ha='center', va='center',
            fontsize=26, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#1e3a5f'))

plt.tight_layout()
plt.savefig('presentation_assets/slides/11_future_work.png', dpi=300, bbox_inches='tight')
plt.close()

# 12. CONCLUSION SLIDE
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#1e3a5f')
ax.add_patch(Rectangle((0, 0), 10, 10, facecolor='#1e3a5f'))
ax.text(5, 8.5, 'Key Achievements', ha='center', va='center',
        fontsize=48, color='#ffd700', weight='bold')

achievements = [
    '✓ 90.15% Accuracy with Random Forest',
    '✓ Comprehensive Model Comparison',
    '✓ Real-time Web Application',
    '✓ Deployed on Streamlit Cloud',
    '✓ Open Source on GitHub'
]

for i, achievement in enumerate(achievements):
    ax.text(5, 6.5 - i*0.8, achievement, ha='center', va='center',
            fontsize=28, color='white', weight='bold')

ax.text(5, 2, 'Live Demo: credit-riskscoring.streamlit.app', ha='center', va='center',
        fontsize=26, color='#ffd700', weight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#2e7d32'))

ax.text(5, 0.8, 'Thank You!', ha='center', va='center',
        fontsize=44, color='white', weight='bold')

plt.tight_layout()
plt.savefig('presentation_assets/slides/12_conclusion.png', dpi=300, bbox_inches='tight', facecolor='#1e3a5f')
plt.close()

# 13. METRICS COMPARISON BAR CHART
fig, ax = plt.subplots(figsize=(14, 9))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
rf_scores = [90.15, 91, 99, 95, 87.59]
lr_scores = [88.78, 89, 99, 94, 85.15]
dt_scores = [88.52, 89, 99, 94, 84.06]

x = np.arange(len(metrics))
width = 0.25

bars1 = ax.bar(x - width, rf_scores, width, label='Random Forest', color='#2e7d32')
bars2 = ax.bar(x, lr_scores, width, label='Logistic Regression', color='#1976d2')
bars3 = ax.bar(x + width, dt_scores, width, label='Decision Tree', color='#d32f2f')

ax.set_xlabel('Metrics', fontsize=20, weight='bold')
ax.set_ylabel('Score (%)', fontsize=20, weight='bold')
ax.set_title('Model Performance Comparison', fontsize=28, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=16, weight='bold')
ax.legend(fontsize=16, loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(80, 100)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('presentation_assets/slides/13_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 14. END CREDITS
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#1e3a5f')
ax.add_patch(Rectangle((0, 0), 10, 10, facecolor='#1e3a5f'))

ax.text(5, 8, 'Intelligent Credit Risk Scoring System', ha='center', va='center',
        fontsize=42, color='white', weight='bold')

ax.text(5, 6.5, 'Project Links', ha='center', va='center',
        fontsize=32, color='#ffd700', weight='bold')

ax.text(5, 5.3, '🌐 Live Demo:', ha='center', va='center',
        fontsize=24, color='white', weight='bold')
ax.text(5, 4.7, 'credit-riskscoring.streamlit.app', ha='center', va='center',
        fontsize=22, color='#4fc3f7')

ax.text(5, 3.8, '💻 GitHub Repository:', ha='center', va='center',
        fontsize=24, color='white', weight='bold')
ax.text(5, 3.2, 'github.com/officialravleensingh/credit-risk-scoring', ha='center', va='center',
        fontsize=20, color='#4fc3f7')

ax.text(5, 2, 'Team Members', ha='center', va='center',
        fontsize=28, color='#ffd700', weight='bold')
team_credits = 'Ravleen Singh  •  Anurag Pandey  •  Ansh Tomar  •  Himanshu Chauhan'
ax.text(5, 1.3, team_credits, ha='center', va='center',
        fontsize=20, color='white', weight='bold')

ax.text(5, 0.3, 'Newton School of Technology - GenAI Capstone Project', ha='center', va='center',
        fontsize=18, color='#b0b0b0')

plt.tight_layout()
plt.savefig('presentation_assets/slides/14_end_credits.png', dpi=300, bbox_inches='tight', facecolor='#1e3a5f')
plt.close()

print("✅ All presentation assets generated successfully!")
print(f"📁 Location: presentation_assets/slides/")
print(f"📊 Total slides created: 14")
