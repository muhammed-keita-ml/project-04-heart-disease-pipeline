# Reproducible End-to-End ML Pipelines for Clinical Risk Prediction:
# Design, Experimentation, and Deployment

**Author:** Muhammed Keita
**Date:** 2026
**Repository:** https://github.com/muhammed-keita-ml/project-04-heart-disease-pipeline
**Live System:** https://huggingface.co/spaces/muhammed-keita-ml/heart-disease-predictor
**Experiment Tracking:** https://dagshub.com/muhammed-keita-ml/heart-disease-pipeline

---

## Abstract

Heart disease remains a leading cause of mortality globally, and early risk
prediction from clinical measurements presents a tractable ML problem with
direct screening applications. This report documents the design, systematic
experimentation, and production deployment of an end-to-end ML pipeline for
binary heart disease classification across 920 patients from four hospital
sites. Four algorithm families were evaluated under consistent preprocessing
and hyperparameter search protocols. The final Random Forest model achieved
AUC=0.921 and Recall=0.922 on the held-out test set. All 40 experiment runs
are fully logged in MLflow on DagsHub. The deployed inference endpoint serves
live predictions via Hugging Face Spaces with automated CI/CD via GitHub
Actions. A notable finding — that GridSearchCV tuning reduced AUC relative to
the baseline Random Forest — motivates further research into search space
design and cross-validation strategy for small structured datasets. This work
raises open questions about distribution shift robustness and monitoring
methodology that motivate ongoing research in Project 06.

---

## 1. Introduction

### 1.1 Problem Context

Heart disease risk prediction from structured clinical measurements is a
well-studied problem with a tractable feature space and well-understood
evaluation criteria. The clinical cost asymmetry between false negatives
(missed disease) and false positives (unnecessary follow-up) makes it a
useful domain for studying threshold-aware model selection and
recall-prioritised evaluation.

The dataset used here aggregates patient records from four hospital sites
(Cleveland, Hungarian, Swiss, VA Long Beach), introducing cross-site
heterogeneity that makes generalisation evaluation non-trivial and motivates
future work on distribution shift robustness.

### 1.2 Engineering Question Addressed

This project addresses the question: what does a rigorous, reproducible model
selection process look like when you must justify your choice — and when the
deployed system must be maintainable, versioned, and auditable?

### 1.3 Contributions

- A fully reproducible end-to-end pipeline from raw data ingestion through
  production deployment
- Systematic comparison of four algorithm families under consistent
  experimental conditions with 40 tracked runs
- Full experiment logging with parameter tracking, metric recording, and
  artifact versioning via MLflow and DagsHub
- A production inference endpoint with automated CI/CD deployment via
  GitHub Actions to Hugging Face Spaces
- The unexpected finding that baseline Random Forest outperformed the
  GridSearchCV-tuned variant — with analysis of why this may have occurred
- Identification of open research questions around distribution shift
  and monitoring in deployed clinical classifiers

---

## 2. Related Work

- **Rajpurkar et al. (2022).** AI in health and medicine. *Nature Medicine.*
  Frames the deployment and reliability challenges for ML in clinical
  decision support — motivates the production-first architecture here.
  Key insight: clinical ML systems require not just high accuracy but
  maintainability, auditability, and monitoring after deployment.

- **Sculley et al. (2015).** Hidden Technical Debt in Machine Learning
  Systems. *NeurIPS.* Motivates the pipeline modularity, experiment
  tracking, and separation of concerns implemented in this system.
  Their taxonomy of ML-specific technical debt informed the decision
  to separate data validation, preprocessing, training, and serving
  into distinct pipeline stages.

- **Grinsztajn et al. (2022).** Why tree-based models still outperform
  deep learning on tabular data. *NeurIPS.* Provides theoretical and
  empirical context for the model selection result — tree-based ensemble
  outperforming other approaches on structured clinical data with under
  10K samples is consistent with their findings across 45 tabular
  datasets.

- **Dua, D. and Graff, C. (2019).** UCI Machine Learning Repository.
  University of California, School of Information and Computer Science.
  Source of the heart disease dataset used in this work.

---

## 3. Methodology

### 3.1 Dataset

- **Source:** UCI Heart Disease Dataset — Cleveland, Hungarian, Swiss,
  and VA Long Beach sites combined
- **Size:** 920 patients, 13 features after preprocessing
- **Target:** Binary classification — heart disease present (1) / absent (0)
- **Cross-site structure:** Four hospital sites with different data
  collection protocols — introduces real-world heterogeneity

### 3.2 Preprocessing Decisions

- **Missing value handling:** Rows with missing target values dropped;
  numeric feature nulls imputed with column mean; categorical nulls
  imputed with mode
- **Feature encoding:** Categorical features one-hot encoded
- **Train/test split:** 80/20 stratified on target to preserve class
  distribution in both sets
- **Scaling:** StandardScaler applied to numeric features before
  Logistic Regression; tree-based models trained on unscaled features
- **Reproducibility:** Random state fixed at 42 across all runs

### 3.3 Model Selection

Four algorithm families evaluated:

1. **Logistic Regression** — linear baseline; interpretable coefficients;
   regularisation via C parameter
2. **Random Forest (baseline)** — tree ensemble with default
   scikit-learn hyperparameters
3. **RF Tuned** — Random Forest with GridSearchCV hyperparameter search
4. **XGBoost** — sequential gradient boosting ensemble

All models evaluated on the same held-out test set under consistent
preprocessing. Primary selection metric: AUC. Tiebreaker: Recall,
reflecting clinical cost asymmetry (false negatives — missed disease —
carry higher cost than false positives in risk screening contexts).

### 3.4 Experiment Tracking

All 40 runs logged to MLflow tracking server hosted on DagsHub.

- **Parameters logged:** algorithm family, hyperparameter values,
  random state, max_iter (where applicable)
- **Metrics logged:** roc_auc, recall, precision, f1_score, accuracy
- **Artifacts logged:** trained model, confusion matrix, ROC curve
- **Reproducibility:** Random state fixed at 42 across all runs;
  all preprocessing steps applied identically across algorithm families

---

## 4. Results

| Model | AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Random Forest (baseline) | **0.921** | **0.922** | 0.847 | 0.883 | 86.4% |
| RF Tuned (GridSearchCV) | 0.915 | 0.912 | 0.823 | 0.865 | 84.2% |
| XGBoost | 0.888 | 0.912 | 0.838 | 0.873 | 85.3% |
| Logistic Regression | 0.903 | 0.882 | 0.841 | 0.861 | 84.2% |

**Deployed model:** Random Forest baseline — selected on highest AUC (0.921)
and Recall (0.922) under consistent experimental conditions.

### 4.1 Key Findings

- The baseline Random Forest outperformed the tuned variant on both AUC
  (0.921 vs 0.915) and Recall (0.922 vs 0.912). GridSearchCV tuning did
  not improve the best metric — see Section 5.3 for analysis.

- XGBoost achieved competitive Recall (0.912) matching RF Tuned, but
  lower AUC (0.888) — indicating stronger performance near the default
  classification threshold but weaker overall discrimination across
  all thresholds.

- Logistic Regression achieved surprisingly strong AUC (0.903),
  outperforming XGBoost on that metric despite being a linear model.
  This suggests the dataset is meaningfully linearly separable — the
  features carry sufficient signal for a linear decision boundary to
  generalise well.

- All four models achieved Recall above 0.88 — the clinical priority
  metric — confirming the dataset supports high-sensitivity classifiers
  across algorithm families. This reduces the model selection decision
  to AUC as the primary discriminator.

---

## 5. Discussion

### 5.1 Interpretation

The Random Forest result is consistent with Grinsztajn et al. (2022) on
tree-based model performance for structured tabular data at this scale.
The ensemble approach outperformed Logistic Regression on AUC by 1.8
percentage points — meaningful but not dramatic, suggesting a dataset
that is tractable for linear approaches but benefits from non-linear
modelling.

The strong Logistic Regression AUC (0.903) is the most practically
significant finding for the clinical context: a fully interpretable
linear model with auditable coefficients achieves near-competitive
discrimination. In a real clinical deployment, interpretability
constraints might favour Logistic Regression despite the AUC gap.

### 5.2 Limitations

- **Dataset scale:** 920 samples from 4 sites. Performance on populations
  with substantially different demographic distributions has not been
  evaluated. Cross-site heterogeneity in the training data may mask
  within-site overfitting.

- **Static training set:** No monitoring for covariate or concept drift
  is implemented at the model layer. Production deployment requires
  integration with a monitoring system — this is addressed in Project 05.

- **Clinical validity:** Recall of 0.922 on the held-out test set should
  not be interpreted as clinical validity. This system has not been
  evaluated in a clinical decision support context and is not intended
  for clinical use.

- **Causal interpretation:** Feature importance is reported but causal
  interpretation is not warranted from this observational dataset.
  Correlation with disease presence does not establish causation.

- **Temporal validation:** The train/test split was random, not
  temporally stratified. In a real deployment scenario, temporal
  validation would be required to assess performance under distribution
  shift over time.

### 5.3 Unexpected Findings

GridSearchCV hyperparameter tuning reduced AUC from 0.921 to 0.915
compared to the baseline Random Forest. This is a counterintuitive
result — tuning is expected to improve or maintain performance, not
reduce it.

Three plausible explanations:

1. **Search space misspecification:** The hyperparameter grid may not
   have included the region containing optimal parameters, causing the
   search to converge on a suboptimal configuration.

2. **Cross-validation variance:** With 920 samples and 5-fold CV, each
   validation fold contains ~184 samples. High variance in fold-level
   performance estimates may have caused the search to select
   hyperparameters that overfit the CV splits rather than generalising
   to the test set.

3. **Default hyperparameters already near-optimal:** scikit-learn's
   default Random Forest parameters (n_estimators=100, no max_depth
   constraint) may already be well-suited to datasets of this scale
   and dimensionality, leaving little room for improvement through
   standard grid search.

This finding motivates more careful search space design in future
experiments — specifically, expanding the grid to cover a wider range
of n_estimators and max_depth values, and using nested cross-validation
to obtain less biased performance estimates.

---

## 6. Future Work and Research Directions

1. **Distribution shift robustness:** Evaluate performance under
   simulated distribution shift (demographic subgroups, temporal splits)
   to characterise generalisation boundaries. This directly motivates
   the monitoring system built in Project 05, which tracks statistical
   drift metrics in real time for this deployed model.

2. **Monitoring integration:** Connect the Hugging Face inference
   endpoint to the Project 05 monitoring API to study how statistical
   drift metrics (PSI, KS test) correlate with real performance
   degradation on this specific system — a concrete instance of the
   broader research question motivating Project 06.

3. **Uncertainty quantification:** Implement conformal prediction
   intervals to provide calibrated confidence estimates alongside
   point predictions. In a risk screening context, knowing when the
   model is uncertain is as clinically valuable as the prediction itself.

4. **Improved hyperparameter search:** Redesign the tuning experiment
   with a wider search space and nested cross-validation to address
   the unexpected finding in Section 5.3. Bayesian optimisation
   (via Optuna or scikit-optimize) would be more efficient than
   grid search at this dataset scale.

5. **Cross-site generalisation:** Train on three sites, test on the
   fourth — repeated for each site. This would provide a more realistic
   estimate of generalisation performance across hospital populations
   and surface the degree of site-specific overfitting in the current
   model.

---

## References

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository.
Irvine, CA: University of California, School of Information and
Computer Science. http://archive.ics.uci.edu/ml

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why tree-based
models still outperform deep learning on tabular data. *Advances in
Neural Information Processing Systems, 35,* 507–520.

Rajpurkar, P., Chen, E., Banerjee, O., & Topol, E. J. (2022).
AI in health and medicine. *Nature Medicine, 28*(1), 31–38.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T.,
Ebner, D., ... & Dennison, D. (2015). Hidden technical debt in
machine learning systems. *Advances in Neural Information Processing
Systems, 28.*
