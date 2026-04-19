# 🎬 ANNOTATED VIDEO SCRIPT WITH SLIDE CUES
## Intelligent Credit Risk Scoring System - 5 Minute Presentation

---

## 🎥 SEGMENT 1: OPENING (0:00 - 0:30)

**👤 SPEAKER:** Ravleen Singh  
**📊 DISPLAY:** `01_title_slide.png`  
**⏱️ DURATION:** 30 seconds

### SCRIPT:
"Hello everyone! Welcome to our presentation on the Intelligent Credit Risk Scoring System. I'm Ravleen Singh, and with me are Anurag Pandey, Ansh Tomar, and Himanshu Chauhan. This is our Milestone 1 project for the GenAI Capstone at Newton School of Technology."

**🎬 DIRECTION:**
- Keep title slide visible entire 30 seconds
- Speak with enthusiasm and energy
- Smile if on camera
- Pause 2 seconds after finishing

---

## 🎥 SEGMENT 2: PROBLEM STATEMENT (0:30 - 1:00)

**👤 SPEAKER:** Ravleen Singh  
**📊 DISPLAY:** `02_problem_statement.png`  
**⏱️ DURATION:** 30 seconds

### SCRIPT:
"Financial institutions face significant challenges with traditional loan evaluation processes. Manual credit assessments are slow, taking days or even weeks. They're also prone to human bias and inconsistent evaluation criteria across different loan officers.

[PAUSE 1 second]

Our project addresses these challenges by building an automated credit risk scoring system that predicts loan repayment probability using machine learning, achieving **90.15% accuracy**. This enables faster, more reliable, and unbiased lending decisions."

**🎬 DIRECTION:**
- Emphasize "90.15% accuracy" 
- Point to or highlight the "Our Solution" section
- Confident, problem-solving tone

---

## 🎥 SEGMENT 3: DATASET & EDA (1:00 - 1:45)

**👤 SPEAKER:** Ansh Tomar  
**📊 DISPLAY:** `03_dataset_overview.png` → then switch to `eda.ipynb`  
**⏱️ DURATION:** 45 seconds

### SCRIPT:
"Let me walk you through our dataset and exploratory data analysis. We worked with a comprehensive credit risk dataset containing **20,000 loan applications** with **21 input features** and 1 target variable - loan paid back.

[KEEP SLIDE 03 VISIBLE - 15 seconds]

The dataset includes four main categories: Demographics, Financial information, Loan details, and Credit history.

[SWITCH TO JUPYTER NOTEBOOK - eda.ipynb]

Through our exploratory data analysis, we discovered that **credit score is the strongest predictor** of loan repayment, followed by delinquency history and debt-to-income ratio. The dataset showed a class distribution of 79.99% paid back versus 20.01% defaulted, and importantly, there were **no missing values**."

**🎬 DIRECTION:**
- Show correlation heatmap in Jupyter notebook
- Highlight credit score correlation
- Zoom in on key visualizations if possible

---

## 🎥 SEGMENT 4: DATA PREPROCESSING (1:45 - 2:20)

**👤 SPEAKER:** Anurag Pandey  
**📊 DISPLAY:** `08_preprocessing_pipeline.png`  
**⏱️ DURATION:** 35 seconds

### SCRIPT:
"I handled the data preprocessing and feature engineering. Our preprocessing pipeline consists of three main steps:

[POINT TO STEP 1]
First, we applied **Label Encoding** to six categorical features - gender, marital status, education level, employment status, loan purpose, and grade subgrade.

[POINT TO STEP 2]
Second, we used **Standard Scaling** on all numerical features to normalize the data, ensuring features with different scales contribute equally to the model.

[POINT TO STEP 3]
Finally, we split the data into **80% training and 20% testing** sets using stratification to maintain the class distribution. This gave us 16,000 training samples and 4,000 testing samples."

**🎬 DIRECTION:**
- Follow the arrow flow on the slide
- Emphasize the three steps clearly
- Optional: Show `utils/preprocessing.py` code snippet

---

## 🎥 SEGMENT 5: MODEL COMPARISON & RESULTS (2:20 - 3:15)

**👤 SPEAKER:** Ravleen Singh  
**📊 DISPLAY:** Multiple slides in sequence  
**⏱️ DURATION:** 55 seconds

### SCRIPT:

**[SLIDE: 04_model_comparison.png - 20 seconds]**
"We evaluated **three machine learning algorithms**: Logistic Regression, Decision Tree, and Random Forest. This comprehensive comparison allowed us to select the best model for our credit risk prediction task.

**[SLIDE: 13_metrics_comparison.png - 10 seconds]**
The results show that **Random Forest outperformed** the other models:
- Random Forest: **90.15% accuracy**, 0.8759 ROC-AUC
- Logistic Regression: 88.78% accuracy, 0.8515 ROC-AUC
- Decision Tree: 88.52% accuracy, 0.8406 ROC-AUC

**[SLIDE: 05_confusion_matrix_rf.png - 15 seconds]**
The Random Forest confusion matrix shows exceptional performance with **99% recall** for paid-back loans, meaning we correctly identify 99% of borrowers who will repay. Out of 4,000 test samples, we correctly predicted 3,181 paid-back loans and 425 defaults.

**[SLIDE: 06_roc_curves.png - 10 seconds]**
The ROC curves demonstrate that Random Forest provides the **best discrimination capability** across all threshold values, making it our model of choice for deployment."

**🎬 DIRECTION:**
- Smooth transitions between slides
- Point to key numbers on each slide
- Emphasize "99% recall" and "best discrimination"
- Keep each slide visible for stated duration

---

## 🎥 SEGMENT 6: FEATURE IMPORTANCE (3:15 - 3:40)

**👤 SPEAKER:** Ansh Tomar  
**📊 DISPLAY:** `07_feature_importance.png`  
**⏱️ DURATION:** 25 seconds

### SCRIPT:
"One of the advantages of Random Forest is its ability to provide **feature importance scores**. Our analysis revealed the top predictors:

[POINT TO TOP BAR]
**Credit score ranks highest**, followed by delinquency history, debt-to-income ratio, and annual income. 

These insights align with our EDA findings and provide **transparency** into how the model makes decisions. This interpretability is crucial for building trust with financial institutions and regulatory compliance."

**🎬 DIRECTION:**
- Highlight the top 4 green bars
- Emphasize "Credit score" as #1
- Professional, confident tone

---

## 🎥 SEGMENT 7: WEB APPLICATION DEMO (3:40 - 4:20)

**👤 SPEAKER:** Himanshu Chauhan  
**📊 DISPLAY:** Live Streamlit Application (`app.py`)  
**⏱️ DURATION:** 40 seconds

### SCRIPT:
"I developed the user interface using **Streamlit**, which provides an intuitive web-based interface for credit risk assessment.

[SHOW APP INTERFACE]
The application is organized into **four input sections**: Personal Information, Financial Information, Loan Details, and Credit History.

[SHOW SIDEBAR]
The sidebar displays our model performance metrics, including the confusion matrix and ROC curve, providing transparency to users.

[START ENTERING DATA]
Let me demonstrate with a sample borrower. I'll enter a 35-year-old employed individual with a credit score of 720, annual income of $60,000, and requesting a $20,000 loan for debt consolidation.

[CLICK 'ASSESS CREDIT RISK' BUTTON]
The system processes the input in **real-time** using our Random Forest model and provides a prediction - in this case, **'Low Risk' with 92% probability** of repayment.

[SHOW URL]
The application is deployed on Streamlit Cloud and accessible worldwide at **credit-riskscoring.streamlit.app**."

**🎬 DIRECTION:**
- Screen record the live application
- Fill in form fields smoothly (can speed up in editing)
- Highlight the prediction result
- Show the URL clearly at the end

**📝 DEMO DATA:**
```
Age: 35
Gender: Male
Marital Status: Married
Education: Bachelor's
Employment: Employed
Annual Income: $60,000
Credit Score: 720
Debt-to-Income: 0.15
Loan Amount: $20,000
Loan Purpose: Debt consolidation
Interest Rate: 10%
Loan Term: 36 months
Delinquency History: 0
Public Records: 0
```

---

## 🎥 SEGMENT 8: TECHNICAL IMPLEMENTATION (4:20 - 4:40)

**👤 SPEAKER:** Anurag Pandey  
**📊 DISPLAY:** `09_project_structure.png` → GitHub repository  
**⏱️ DURATION:** 20 seconds

### SCRIPT:

**[SLIDE: 09_project_structure.png - 10 seconds]**
"Our project follows a **modular structure** with clear separation of concerns:
- app.py contains the Streamlit web application
- train_model.py trains the Random Forest model and generates visualizations
- compare_models.py compares all three algorithms
- utils/preprocessing.py includes all data preprocessing functions
- models/model_params.py stores the trained model parameters
- visualizations/ folder contains all generated charts and graphs

**[SWITCH TO GITHUB - 10 seconds]**
All code is **version-controlled on GitHub** with meaningful commit history, making collaboration seamless across our team."

**🎬 DIRECTION:**
- Show file structure clearly
- Switch to GitHub repository view
- Show commits and file organization
- Professional, technical tone

---

## 🎥 SEGMENT 9: CHALLENGES & SOLUTIONS (4:40 - 4:55)

**👤 SPEAKER:** Ansh Tomar  
**📊 DISPLAY:** `10_challenges_solutions.png`  
**⏱️ DURATION:** 15 seconds

### SCRIPT:
"We encountered several challenges during development. The **class imbalance** of 80-20 was addressed using stratified splitting. **Feature scaling** differences were resolved with StandardScaler. **Model selection** was optimized through comprehensive comparison of three algorithms. And **deployment issues** with binary model files were solved by converting to Python module format."

**🎬 DIRECTION:**
- Speak quickly but clearly (15 seconds is tight)
- Point to each challenge-solution pair
- Confident problem-solving tone

---

## 🎥 SEGMENT 10: CONCLUSION & FUTURE WORK (4:55 - 5:30)

**👤 SPEAKER:** Ravleen Singh  
**📊 DISPLAY:** `11_future_work.png` → `12_conclusion.png`  
**⏱️ DURATION:** 35 seconds

### SCRIPT:

**[SLIDE: 11_future_work.png - 15 seconds]**
"To conclude, we successfully built an automated credit risk scoring system that:
- Achieves **90.15% accuracy** with Random Forest
- Compared three machine learning algorithms systematically
- Provides real-time predictions through an intuitive web interface
- Includes comprehensive visualizations for model interpretability
- Is deployed and accessible worldwide

**[SLIDE: 12_conclusion.png - 20 seconds]**
For future work, we plan to implement advanced ensemble models like **XGBoost**, add **SHAP values** for individual prediction explanations, and develop a **REST API** for integration with banking systems.

Thank you for watching! Our project is **open-source** and available on GitHub. Feel free to check out the live demo at **credit-riskscoring.streamlit.app**."

**🎬 DIRECTION:**
- Enthusiastic, concluding tone
- Emphasize key achievements
- Smile and show confidence
- Clear call-to-action with URL

---

## 🎥 SEGMENT 11: CLOSING (5:30 - 5:40)

**👤 SPEAKER:** All Contributors (Ravleen, Anurag, Ansh, Himanshu)  
**📊 DISPLAY:** `14_end_credits.png`  
**⏱️ DURATION:** 10 seconds

### SCRIPT:
**[ALL TOGETHER]**
"Thank you!"

**🎬 DIRECTION:**
- All team members say "Thank you!" in unison
- Can be recorded separately and synced in editing
- Keep end credits visible for full 10 seconds
- Fade to black

---

## 📋 POST-RECORDING CHECKLIST

### For Each Speaker:
- [ ] Audio is clear and audible
- [ ] No background noise or interruptions
- [ ] Slides are visible and readable
- [ ] Timing matches the script (±5 seconds okay)
- [ ] Transitions are smooth
- [ ] File saved with clear name (e.g., "Ravleen_Opening.mp4")

### For Editor:
- [ ] All segments received
- [ ] Audio levels normalized
- [ ] Transitions added between speakers
- [ ] Text overlays added for key metrics
- [ ] Background music added (optional)
- [ ] Total duration: 5:00 - 5:40 minutes
- [ ] Exported in 1080p MP4

---

## 🎯 KEY METRICS TO EMPHASIZE

Throughout the video, these numbers should be highlighted:

- **20,000** samples
- **21** features
- **90.15%** accuracy
- **0.8759** ROC-AUC
- **99%** recall
- **80-20** split
- **3** algorithms compared

---

## 🔗 URLS TO DISPLAY

- **Live Demo:** credit-riskscoring.streamlit.app
- **GitHub:** github.com/officialravleensingh/credit-risk-scoring

---

## 💡 FINAL TIPS

1. **Practice makes perfect** - Read your section 3-4 times before recording
2. **Speak naturally** - Don't sound like you're reading
3. **Pause between points** - Gives editor room to work
4. **Smile** - Even if not on camera, it affects your voice
5. **Stay hydrated** - Keep water nearby
6. **Take breaks** - If you mess up, pause, breathe, and restart
7. **Have fun!** - Your enthusiasm will show

---

**You've got this! 🎉 Good luck with your recording! 🚀**
