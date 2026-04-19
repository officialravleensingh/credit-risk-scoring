# Video Production Guide - Credit Risk Scoring System
## Complete Slide-to-Script Mapping

---

## 📁 Generated Assets Location
**Folder:** `presentation_assets/slides/`
**Total Slides:** 14 high-resolution PNG images (300 DPI)

---

## 🎬 COMPLETE TIMELINE WITH SLIDE MAPPING

### **SEGMENT 1: OPENING (0:00 - 0:30) - Ravleen Singh**

**📊 Slide to Display:** `01_title_slide.png`

**Script:**
> "Hello everyone! Welcome to our presentation on the Intelligent Credit Risk Scoring System. I'm Ravleen Singh, and with me are Anurag Pandey, Ansh Tomar, and Himanshu Chauhan. This is our Milestone 1 project for the GenAI Capstone at Newton School of Technology."

**Visual Elements:**
- Project title in large bold text
- All 4 team member names
- Subtitle: "ML-Based Loan Repayment Prediction"
- Footer: Newton School of Technology branding

---

### **SEGMENT 2: PROBLEM STATEMENT (0:30 - 1:00) - Ravleen Singh**

**📊 Slide to Display:** `02_problem_statement.png`

**Script:**
> "Financial institutions face significant challenges with traditional loan evaluation processes. Manual credit assessments are slow, taking days or even weeks. They're also prone to human bias and inconsistent evaluation criteria across different loan officers. Our project addresses these challenges by building an automated credit risk scoring system that predicts loan repayment probability using machine learning, achieving 90.15% accuracy. This enables faster, more reliable, and unbiased lending decisions."

**Visual Elements:**
- 4 key challenges with icons (time, bias, inconsistency, errors)
- "Our Solution" section highlighting automation
- Key metrics: 90.15% accuracy, real-time, unbiased

---

### **SEGMENT 3: DATASET & EDA (1:00 - 1:45) - Ansh Tomar**

**📊 Slides to Display:** 
1. `03_dataset_overview.png` (1:00 - 1:30)
2. Switch to Jupyter Notebook `eda.ipynb` (1:30 - 1:45)

**Script:**
> "Let me walk you through our dataset and exploratory data analysis. We worked with a comprehensive credit risk dataset containing 20,000 loan applications with 21 input features and 1 target variable - loan paid back. The dataset includes four main categories: Demographics, Financial information, Loan details, and Credit history. Through our exploratory data analysis, we discovered that credit score is the strongest predictor of loan repayment, followed by delinquency history and debt-to-income ratio. The dataset showed a class distribution of 79.99% paid back versus 20.01% defaulted, and importantly, there were no missing values."

**Visual Elements:**
- Dataset statistics table (20,000 samples, 21 features)
- Class distribution percentages
- 4 feature categories highlighted
- **Switch to:** Live Jupyter notebook showing correlation heatmap

---

### **SEGMENT 4: DATA PREPROCESSING (1:45 - 2:20) - Anurag Pandey**

**📊 Slide to Display:** `08_preprocessing_pipeline.png`

**Script:**
> "I handled the data preprocessing and feature engineering. Our preprocessing pipeline consists of three main steps: First, we applied Label Encoding to six categorical features - gender, marital status, education level, employment status, loan purpose, and grade subgrade. Second, we used Standard Scaling on all numerical features to normalize the data, ensuring features with different scales contribute equally to the model. Finally, we split the data into 80% training and 20% testing sets using stratification to maintain the class distribution. This gave us 16,000 training samples and 4,000 testing samples."

**Visual Elements:**
- 4-step pipeline with arrows
- Raw Data → Label Encoding → Standard Scaling → Train-Test Split
- Visual flow diagram

**Optional:** Show `utils/preprocessing.py` code snippet in VS Code

---

### **SEGMENT 5: MODEL COMPARISON & RESULTS (2:20 - 3:15) - Ravleen Singh**

**📊 Slides to Display (in sequence):**
1. `04_model_comparison.png` (2:20 - 2:40)
2. `13_metrics_comparison.png` (2:40 - 2:50)
3. `05_confusion_matrix_rf.png` (2:50 - 3:05)
4. `06_roc_curves.png` (3:05 - 3:15)

**Script:**
> "We evaluated three machine learning algorithms: Logistic Regression, Decision Tree, and Random Forest. This comprehensive comparison allowed us to select the best model for our credit risk prediction task. The results show that Random Forest outperformed the other models: Random Forest: 90.15% accuracy, 0.8759 ROC-AUC; Logistic Regression: 88.78% accuracy, 0.8515 ROC-AUC; Decision Tree: 88.52% accuracy, 0.8406 ROC-AUC. The Random Forest confusion matrix shows exceptional performance with 99% recall for paid-back loans, meaning we correctly identify 99% of borrowers who will repay. Out of 4,000 test samples, we correctly predicted 3,181 paid-back loans and 425 defaults. The ROC curves demonstrate that Random Forest provides the best discrimination capability across all threshold values, making it our model of choice for deployment."

**Visual Elements:**
- Comparison table with all 5 metrics
- Bar chart comparing models
- Confusion matrix with actual numbers
- ROC curves overlaid for all 3 models

---

### **SEGMENT 6: FEATURE IMPORTANCE (3:15 - 3:40) - Ansh Tomar**

**📊 Slide to Display:** `07_feature_importance.png`

**Script:**
> "One of the advantages of Random Forest is its ability to provide feature importance scores. Our analysis revealed the top predictors: Credit score ranks highest, followed by delinquency history, debt-to-income ratio, and annual income. These insights align with our EDA findings and provide transparency into how the model makes decisions. This interpretability is crucial for building trust with financial institutions and regulatory compliance."

**Visual Elements:**
- Horizontal bar chart showing top 8 features
- Credit Score (0.28) highlighted as most important
- Color-coded bars (top 4 in green)

---

### **SEGMENT 7: WEB APPLICATION (3:40 - 4:20) - Himanshu Chauhan**

**📊 Screen Recording:** Live Streamlit Application (`app.py`)

**Script:**
> "I developed the user interface using Streamlit, which provides an intuitive web-based interface for credit risk assessment. The application is organized into four input sections: Personal Information, Financial Information, Loan Details, and Credit History. The sidebar displays our model performance metrics, including the confusion matrix and ROC curve, providing transparency to users. Let me demonstrate with a sample borrower. I'll enter a 35-year-old employed individual with a credit score of 720, annual income of $60,000, and requesting a $20,000 loan for debt consolidation. The system processes the input in real-time using our Random Forest model and provides a prediction - in this case, 'Low Risk' with 92% probability of repayment. The application is deployed on Streamlit Cloud and accessible worldwide at credit-riskscoring.streamlit.app."

**Demo Input Values:**
- Age: 35
- Gender: Male
- Marital Status: Married
- Education: Bachelor's
- Employment: Employed
- Annual Income: $60,000
- Credit Score: 720
- Debt-to-Income: 0.15
- Loan Amount: $20,000
- Loan Purpose: Debt consolidation
- Interest Rate: 10%
- Loan Term: 36 months
- Delinquency History: 0
- Public Records: 0

**Expected Output:** Low Risk, ~92% probability

---

### **SEGMENT 8: TECHNICAL IMPLEMENTATION (4:20 - 4:40) - Anurag Pandey**

**📊 Slides to Display:**
1. `09_project_structure.png` (4:20 - 4:30)
2. GitHub repository view (4:30 - 4:40)

**Script:**
> "Our project follows a modular structure with clear separation of concerns: app.py contains the Streamlit web application, train_model.py trains the Random Forest model and generates visualizations, compare_models.py compares all three algorithms, utils/preprocessing.py includes all data preprocessing functions, models/model_params.py stores the trained model parameters, visualizations/ folder contains all generated charts and graphs. All code is version-controlled on GitHub with meaningful commit history, making collaboration seamless across our team."

**Visual Elements:**
- File tree structure in monospace font
- GitHub repository screenshot showing commits

---

### **SEGMENT 9: CHALLENGES & SOLUTIONS (4:40 - 4:55) - Ansh Tomar**

**📊 Slide to Display:** `10_challenges_solutions.png`

**Script:**
> "We encountered several challenges during development. The class imbalance of 80-20 was addressed using stratified splitting. Feature scaling differences were resolved with StandardScaler. Model selection was optimized through comprehensive comparison of three algorithms. And deployment issues with binary model files were solved by converting to Python module format."

**Visual Elements:**
- 4 challenge-solution pairs
- Challenges in red, solutions in green
- Clean, easy-to-read format

---

### **SEGMENT 10: CONCLUSION & FUTURE WORK (4:55 - 5:30) - Ravleen Singh**

**📊 Slides to Display:**
1. `11_future_work.png` (4:55 - 5:10)
2. `12_conclusion.png` (5:10 - 5:30)

**Script:**
> "To conclude, we successfully built an automated credit risk scoring system that: Achieves 90.15% accuracy with Random Forest, Compared three machine learning algorithms systematically, Provides real-time predictions through an intuitive web interface, Includes comprehensive visualizations for model interpretability, and Is deployed and accessible worldwide. For future work, we plan to implement advanced ensemble models like XGBoost, add SHAP values for individual prediction explanations, and develop a REST API for integration with banking systems. Thank you for watching! Our project is open-source and available on GitHub. Feel free to check out the live demo at credit-riskscoring.streamlit.app."

**Visual Elements:**
- Future enhancements (6 items with icons)
- Key achievements checklist
- Live demo URL prominently displayed

---

### **SEGMENT 11: CLOSING (5:30 - 5:40) - All Contributors**

**📊 Slide to Display:** `14_end_credits.png`

**Script (All Together):**
> "Thank you!"

**Visual Elements:**
- Project title
- Live demo URL
- GitHub repository link
- All team member names
- Newton School branding

---

## 🎥 RECORDING INSTRUCTIONS

### Equipment Setup
- **Screen Recording:** OBS Studio, Loom, or Zoom
- **Microphone:** External USB mic recommended (or quality built-in)
- **Resolution:** 1920x1080 (1080p)
- **Frame Rate:** 30 FPS minimum

### Recording Tips
1. **Individual Segments:** Each person records their section separately
2. **Quiet Environment:** Minimize background noise
3. **Clear Audio:** Speak clearly, not too fast
4. **Slide Timing:** Keep slides visible for at least 5 seconds each
5. **Transitions:** Pause 1 second between slides

### Screen Recording Sections
- **Slides:** Use image viewer or PowerPoint for PNG slides
- **Jupyter Notebook:** Show `notebooks/eda.ipynb` with visualizations
- **VS Code:** Display `utils/preprocessing.py` and project structure
- **Streamlit App:** Live demo at localhost or deployed URL
- **GitHub:** Show repository structure and commits

---

## ✂️ EDITING CHECKLIST

### Video Editing
- [ ] Trim long pauses and "umm/ahh" sounds
- [ ] Add smooth transitions between speakers (fade/dissolve)
- [ ] Normalize audio levels across all segments
- [ ] Add text overlays for key metrics:
  - "90.15% Accuracy"
  - "0.8759 ROC-AUC"
  - "20,000 Samples"
- [ ] Zoom in on important visualizations
- [ ] Add subtle background music (optional, low volume)
- [ ] Include captions/subtitles (optional but recommended)

### Quality Checks
- [ ] Total duration: 5:00 - 5:40 minutes
- [ ] Audio is clear and consistent
- [ ] All slides are visible and readable
- [ ] Transitions are smooth
- [ ] No awkward pauses
- [ ] Export in 1080p MP4 format

---

## 📊 SLIDE REFERENCE GUIDE

| Slide # | Filename | Section | Speaker | Duration |
|---------|----------|---------|---------|----------|
| 1 | 01_title_slide.png | Opening | Ravleen | 0:00-0:30 |
| 2 | 02_problem_statement.png | Problem | Ravleen | 0:30-1:00 |
| 3 | 03_dataset_overview.png | Dataset | Ansh | 1:00-1:30 |
| - | eda.ipynb (live) | EDA | Ansh | 1:30-1:45 |
| 4 | 08_preprocessing_pipeline.png | Preprocessing | Anurag | 1:45-2:20 |
| 5 | 04_model_comparison.png | Models | Ravleen | 2:20-2:40 |
| 6 | 13_metrics_comparison.png | Metrics | Ravleen | 2:40-2:50 |
| 7 | 05_confusion_matrix_rf.png | Results | Ravleen | 2:50-3:05 |
| 8 | 06_roc_curves.png | ROC | Ravleen | 3:05-3:15 |
| 9 | 07_feature_importance.png | Features | Ansh | 3:15-3:40 |
| - | app.py (live demo) | Web App | Himanshu | 3:40-4:20 |
| 10 | 09_project_structure.png | Structure | Anurag | 4:20-4:30 |
| - | GitHub (live) | GitHub | Anurag | 4:30-4:40 |
| 11 | 10_challenges_solutions.png | Challenges | Ansh | 4:40-4:55 |
| 12 | 11_future_work.png | Future | Ravleen | 4:55-5:10 |
| 13 | 12_conclusion.png | Conclusion | Ravleen | 5:10-5:30 |
| 14 | 14_end_credits.png | Closing | All | 5:30-5:40 |

---

## 🎯 KEY TALKING POINTS REMINDER

### Ravleen's Sections
- "90.15% accuracy with Random Forest"
- "Systematic comparison of 3 algorithms"
- "99% recall for paid-back loans"
- "Best discrimination capability"

### Anurag's Sections
- "Label encoding for 6 categorical features"
- "Standard scaling for normalization"
- "80-20 stratified split"
- "Modular project structure"

### Ansh's Sections
- "20,000 loan applications"
- "Credit score is strongest predictor"
- "Feature importance provides transparency"
- "Stratified splitting solved class imbalance"

### Himanshu's Section
- "Four input sections in the app"
- "Real-time predictions"
- "Sidebar shows model metrics"
- "Deployed on Streamlit Cloud"

---

## 📝 POST-PRODUCTION

### File Naming
- Final video: `Credit_Risk_Scoring_Presentation.mp4`
- Backup: `Credit_Risk_Scoring_Presentation_v1.mp4`

### Distribution
- Upload to YouTube (unlisted or public)
- Share link with Newton School
- Include in GitHub README
- Add to project portfolio

### Metadata
- **Title:** Intelligent Credit Risk Scoring System - ML Project Presentation
- **Description:** Automated credit risk assessment using Random Forest achieving 90.15% accuracy. Built with Python, scikit-learn, and Streamlit.
- **Tags:** machine learning, credit risk, random forest, python, streamlit, data science, loan prediction

---

## 🚀 QUICK START GUIDE

1. **Review all 14 slides** in `presentation_assets/slides/`
2. **Practice your section** with the script
3. **Set up recording environment** (quiet room, good lighting)
4. **Record your segment** (aim for 1-2 takes)
5. **Share recordings** with team for editing
6. **Edit together** using video editing software
7. **Review final cut** as a team
8. **Export and upload** to YouTube/platform

---

## 📞 TEAM COORDINATION

### Recording Schedule (Suggested)
- **Day 1:** Ravleen records Opening + Problem Statement
- **Day 2:** Ansh records Dataset + EDA + Feature Importance
- **Day 3:** Anurag records Preprocessing + Technical Implementation
- **Day 4:** Himanshu records Web Application Demo
- **Day 5:** Ravleen records Model Comparison + Conclusion
- **Day 6-7:** Team editing and final review

### File Sharing
- Use Google Drive or Dropbox for raw recordings
- Share folder: `Credit_Risk_Video_Production/`
- Subfolders: `raw_recordings/`, `edited_segments/`, `final_video/`

---

## ✅ FINAL CHECKLIST

- [ ] All 14 slides generated and reviewed
- [ ] Script practiced by all team members
- [ ] Recording equipment tested
- [ ] Jupyter notebook prepared with visualizations
- [ ] Streamlit app running and tested
- [ ] GitHub repository cleaned and organized
- [ ] Demo data prepared (sample borrower profile)
- [ ] All segments recorded
- [ ] Audio quality checked
- [ ] Video edited and transitions added
- [ ] Final video reviewed by all team members
- [ ] Video exported in 1080p
- [ ] Video uploaded and link shared

---

**Good luck with your presentation! 🎉**
