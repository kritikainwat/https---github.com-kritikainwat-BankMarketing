# Bank Marketing — Term Deposit Prediction (Full Report)
Author: Kritika Inwati
Date: 10 -12 2025
Project: Bank Marketing — Machine Learning Pipeline & Streamlit App Deployment  

---

# Abstract
This project analyses the Portuguese Bank Marketing dataset to predict whether a client will subscribe to a term deposit.  
We perform **exploratory data analysis**, **data cleaning**, **feature engineering**, handle **class imbalance with SMOTE**, build and tune several models (Logistic Regression, SVM, Random Forest, XGBoost), evaluate them using multiple metrics, and finally deploy the best model as a **Streamlit web app**.  

Key results:  
- Logistic Regression: Accuracy 82.5%, Recall 78%, ROC-AUC 0.894  
- SVM: Accuracy 85.1%, Recall 77%, ROC-AUC 0.900  
- Random Forest: Accuracy 88.8%, Recall 31%, ROC-AUC 0.899  

The app accepts new client data and returns real-time predictions to support marketing decisions.  

---

# 1. Introduction
Banks often run marketing campaigns to promote term deposits. However, contacting every client is costly and inefficient.  
Machine learning can help by predicting **who is most likely to subscribe**, so campaigns can be better targeted.  

**Objective:** Build an ML pipeline to:  
- Understand the data (EDA)  
- Preprocess & engineer features  
- Handle imbalance with SMOTE  
- Train & evaluate models  
- Deploy best model as an app  

---

# 2. Dataset Description
- Source: UCI Portuguese Bank Marketing Dataset  
- ~41,188 rows × 21 columns  
- Target: `y` (yes = subscribed, no = not subscribed)  

**Examples of features:**  
- `age` (numeric)  
- `job`, `marital`, `education` (categorical)  
- `balance` (numeric, skewed)  
- `pdays` (days since last contact, -1 means never)  
- `duration` (call duration in seconds — warning: leakage feature)  

**Class distribution:**  
- `yes`: ~11%  
- `no`: ~89%  
This imbalance is why **SMOTE** is important.  

---

# 3. Exploratory Data Analysis (EDA)
We start by checking distributions and correlations.  

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("bank.csv")
print(df.shape)
print(df['y'].value_counts(normalize=True))

# Example: Age distribution
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
```

**Insights:**  
- Most clients are between 30–50 years.  
- Balance is highly skewed (many negative/zero values).  
- `duration` strongly correlates with outcome (but should not be used if predicting before call).  
- Class imbalance confirmed: only ~11% subscribed.  

---

# 4. Data Cleaning
Steps taken:  
1. **Missing values:** Not present in dataset.  
2. **Outliers:** Balance outliers clipped.  
3. **pdays:** -1 replaced with new column `pdays_imputed` and `contacted_before`.  

**Why?**  
- If we don’t handle -1 in `pdays`, models treat it as valid numeric value.  
- If we don’t clip balance, extreme outliers dominate model training.  

```python
import numpy as np

df['balance_clipped'] = np.clip(df['balance'], -2000, 20000)
df['balance_log'] = np.log1p(df['balance_clipped'])
df['pdays_imputed'] = df['pdays'].replace(-1, 999)
df['contacted_before'] = (df['pdays'] != -1).astype(int)
```

---

# 5. Feature Engineering
We created:  
- `balance_clipped` (reduce effect of extreme outliers)  
- `balance_log` (normalize skewed balance)  
- `pdays_imputed` (replace -1 with 999)  
- `contacted_before` (binary flag for previous contact)  

**If not done:** Models would misinterpret raw skewed balance and `pdays = -1`.  

---

# 6. Pipelines — Why and How
**Why pipelines?**  
- To avoid repeating preprocessing steps manually.  
- To prevent data leakage during CV.  
- To combine preprocessing + model training in one object.  

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_cols = ["age", "balance_clipped", "balance_log", "day", "duration", "campaign", "pdays_imputed", "previous"]
categorical_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ]
)
```

---

# 7. Handling Imbalance — SMOTE
**Why SMOTE?**  
- Target variable is imbalanced (yes = 11%).  
- Models without balancing predict "No" most of the time (high accuracy but low recall).  
- SMOTE oversamples minority class by creating synthetic examples.  

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```

**What if not used?** Recall for "yes" would be very low.  

---

# 8. Models Tried
We compared multiple models:  

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

log_reg = LogisticRegression(max_iter=500)
svm = SVC(probability=True)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
```

**Strengths:**  
- Logistic Regression → interpretable, fast  
- SVM → strong boundary classifier (needs scaling)  
- Random Forest → captures nonlinear patterns, robust  
- XGBoost → high performance on tabular data  

---

# 9. Evaluation Metrics
Why multiple metrics? Accuracy alone is misleading with imbalance.  

- **Accuracy:** Overall correctness.  
- **Precision:** Of predicted "yes", how many were correct.  
- **Recall:** Of actual "yes", how many were found.  
- **F1:** Balance between precision and recall.  
- **ROC-AUC:** Probability ranking quality.  

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
```

---

# 10. Results Summary

| Model            | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | ROC-AUC |
|------------------|----------|-----------------|--------------|----------|---------|
| Logistic Regression | 0.825 | 0.38 | 0.78 | 0.51 | 0.894 |
| SVM                 | 0.851 | 0.42 | 0.77 | 0.54 | 0.900 |
| Random Forest       | 0.888 | 0.52 | 0.31 | 0.39 | 0.899 |

**Observations:**  
- Random Forest has highest accuracy, but recall is low.  
- Logistic Regression & SVM balance recall better.  
- Depending on business need (catching subscribers), SVM may be preferable.  

---

# 11. Cross-Validation & Hyperparameter Tuning
We used **GridSearchCV** with 5-fold CV:  

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"clf__C": [0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

**Why?** Ensures best parameters chosen without overfitting.  

---

# 12. Final Model & Interpretation
- Chosen model: [state final choice, e.g., SVM or RF depending on recall].  
- Feature importances (RandomForest) or coefficients (Logistic).  

---

# 13. Streamlit App
We deployed a user-friendly app:  

```python
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("best_rf_pipeline.joblib")

st.title("📊 Bank Term Deposit Prediction App")

age = st.number_input("Age", 18, 100, 30)
job = st.selectbox("Job", ["admin.","blue-collar","student","technician","unknown"])
# ... (other inputs) ...

input_df = pd.DataFrame({...})

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.write("Prediction:", pred)
```

Run locally:  
```bash
streamlit run app/app.py
```

---

# 14. GitHub Deployment
**Repo Structure:**  
```
BankMarketingProject/
├── app/
│   └── app.py
├── models/
│   └── best_rf_pipeline.joblib
├── data/
│   └── bank.csv
├── notebooks/
│   └── analysis.ipynb
├── REPORT_FULL.md
└── requirements.txt
```

---

# 15. Limitations & Future Work
- Data is old, may not generalize to modern campaigns.  
- Duration feature leakage.  
- SMOTE creates synthetic data, may not fully reflect reality.  
- Future: try cost-sensitive learning, SHAP explainability, deploy on cloud.  

---

# 16. Conclusion
- Built an end-to-end ML pipeline.  
- Compared models & evaluated with multiple metrics.  
- Deployed a Streamlit app for real-time predictions.  
- Provided GitHub repo for reproducibility.  

---

# Appendix
## Requirements
```
pandas==2.2.2
numpy==1.26.2
scikit-learn==1.6.1
imbalanced-learn==0.11.2
xgboost==1.7.6
streamlit==1.25.0
joblib==1.3.2
matplotlib
seaborn
```

## Example Confusion Matrix Plot
```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.show()
```

---
