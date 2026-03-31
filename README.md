# Project 04 — Heart Disease ML Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end machine learning pipeline predicting heart disease presence from 
clinical measurements. Built with scikit-learn, tracked with MLflow on DagsHub, 
and deployed as an interactive Gradio app on Hugging Face Spaces.

**Live Demo:** [Heart Disease Risk Predictor](https://huggingface.co/spaces/muhammed-keita-ml/heart-disease-predictor)  
**Experiment Tracking:** [DagsHub MLflow](https://dagshub.com/muhammed-keita-ml/heart-disease-pipeline.mlflow)

---

## Problem Statement

Heart disease is the leading cause of death globally. Early detection from routine 
clinical measurements can significantly improve patient outcomes. This project builds 
a binary classifier to predict heart disease presence from 13 clinical features 
collected across 4 hospitals.

---

## Dataset

- **Source:** UCI Heart Disease Dataset (4-hospital combined version)
- **Rows:** 920 patients
- **Features:** 13 clinical measurements
- **Target:** Binary — heart disease present (1) or absent (0)
- **Class balance:** 55.3% positive, 44.7% negative
- **Hospitals:** Cleveland, Hungary, Switzerland, VA Long Beach

### Key Features

| Feature | Description |
|---|---|
| `age` | Age in years |
| `thalch` | Maximum heart rate achieved |
| `oldpeak` | ST depression induced by exercise |
| `exang` | Exercise-induced angina |
| `cp` | Chest pain type |
| `ca` | Number of major vessels coloured by fluoroscopy |
| `thal` | Thalassemia stress test result |

---

## Pipeline Overview
```
Raw Data → Cleaning → EDA → Preprocessing → Training → Evaluation → MLflow → Deployment
```

### Phase 1 — Data Cleaning
- Dropped non-predictive columns (`id`, `dataset`)
- Binarised target: `num` values 1–4 collapsed to 1
- Mode imputation for categorical columns before encoding
- Median imputation for numeric columns
- One-hot encoding for 7 categorical features
- Notable: `ca` (66% missing), `thal` (53% missing) — flagged in analysis

### Phase 2 — EDA
- Target distribution: mildly balanced (55/45)
- Strongest linear predictors: `exang` (r=0.43), `thalch` (r=-0.38), `oldpeak` (r=0.37)
- Key finding: lower max heart rate and higher ST depression strongly indicate disease

### Phase 3 — Preprocessing
- 80/20 stratified train/test split (random_state=42)
- StandardScaler fitted on train set only (no data leakage)

### Phase 4 — Model Training
- 5-fold StratifiedKFold cross-validation
- Models: Logistic Regression, Random Forest, XGBoost
- GridSearchCV hyperparameter tuning on Random Forest

### Phase 5 — Evaluation

| Model | ROC-AUC | Accuracy | Disease Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9032 | 0.84 | 0.88 | 0.86 |
| **Random Forest** | **0.9207** | **0.86** | **0.92** | **0.88** |
| XGBoost | 0.8870 | 0.85 | 0.91 | 0.87 |
| RF Tuned | 0.9146 | 0.84 | 0.91 | 0.87 |

**Best model: Random Forest — ROC-AUC 0.921, Disease Recall 0.92**

The model correctly identifies 92 out of every 100 heart disease patients.

### Phase 6 — MLflow Tracking
- All 4 models logged with parameters, metrics, and artifacts
- Remote tracking server: DagsHub
- Experiment: `heart-disease-project`

### Phase 7 — Deployment
- Gradio app with 13 clinical input controls
- Deployed to Hugging Face Spaces
- Live public demo available

---

## Results

### Cross-Validation (5-fold ROC-AUC)

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression | 0.8839 | 0.0204 |
| Random Forest | 0.8707 | 0.0268 |
| XGBoost | 0.8427 | 0.0278 |

### Key Insight
Logistic Regression led CV scoring, indicating strong linear relationships 
between features and target. Random Forest outperformed on the test set 
after capturing non-linear interactions. The dataset's strong linear 
signals (confirmed by correlation analysis) explain why a simple baseline 
nearly matches complex ensemble methods.

---

## Feature Importance

Top predictors by Random Forest importance:
1. `thalch` — maximum heart rate
2. `oldpeak` — ST depression
3. `exang_True` — exercise-induced angina
4. `age` — patient age
5. `ca` — major vessels coloured

These align with clinical literature on ischemic heart disease markers.

---

## Project Structure
```
project-04-heart-disease-pipeline/
├── app.py                          # Gradio app for HF Spaces
├── requirements.txt                # HF Spaces dependencies
├── heart_disease_model.pkl         # Trained Random Forest model
├── feature_names.json              # Feature column order for inference
├── heart-disease-ml-pipeline.ipynb # Full training notebook
└── README.md                       # This file
```

---

## How to Run Locally
```bash
# Clone the repository
git clone https://github.com/muhammed-keita-ml/project-04-heart-disease-pipeline

# Install dependencies
pip install scikit-learn numpy joblib gradio

# Launch the app
python app.py
```

---

## Tools & Technologies

| Category | Tool |
|---|---|
| Language | Python 3.12 |
| ML Framework | scikit-learn, XGBoost |
| Experiment Tracking | MLflow + DagsHub |
| App Framework | Gradio |
| Deployment | Hugging Face Spaces |
| Notebook | Kaggle |
| Version Control | GitHub |

---

## Author

**Muhammed Keita** — ML Engineer in Training  
[GitHub](https://github.com/muhammed-keita-ml) · 
[LinkedIn](https://linkedin.com/in/muhammed-keita) · 
[Hugging Face](https://huggingface.co/muhammed-keita-ml)

---

*Part of an end-to-end ML/MLOps portfolio. Projects 01–03 available on GitHub.*
