# Credit Risk Scoring System (XGBoost + SHAP + FastAPI + Streamlit)

An end-to-end, industry-style machine learning project that predicts the probability of loan default using Boosting (XGBoost), domain-driven feature engineering, explainable AI (SHAP), a REST API (FastAPI), and an interactive frontend (Streamlit).

This project simulates how real fintech and banking systems evaluate credit applications and assist in loan approval decisions.

---

#  Project Overview

This system predicts whether a customer is likely to default on a loan and generates:

* Default probability
* Risk score (0–100)
* Approval decision (Approve / Review / Reject)
* Top contributing risk factors using SHAP

It includes a full ML pipeline from data preprocessing to model deployment.

---

#  Key Features

* Boosting model using XGBoost
* Domain-driven feature engineering
* Explainable AI with SHAP
* REST API using FastAPI
* Interactive frontend using Streamlit
* Risk scoring system used in banking workflows
* Real-time prediction pipeline
* Training-serving consistency (no skew)

---

#  Business Problem

Financial institutions need to decide:

> "Will this customer default on a loan?"

This system helps:

* Assess creditworthiness
* Assign a risk score
* Categorize applicants into risk groups
* Support automated loan decision pipelines

---

#  Risk Decision Logic

Risk Score Range → Decision

* 70–100 → Approve (Low Risk)
* 40–69 → Review (Medium Risk)
* 0–39 → Reject (High Risk)

---

#  System Architecture

User Input → Streamlit UI → FastAPI → Feature Engineering → XGBoost Model → SHAP → Risk Score + Decision → UI Display


---

#  Project Structure

```
credit-risk-system/
│
├── data/
│   ├── german_credit_data.csv
│   └── processed_data.csv
│
├── notebook/
│   └── eda.ipynb
│
├── models/
│   └── xgboost_model.pkl
│
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── explain_model.py
│   └── predict.py
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
└── README.md
```

---

#  Dataset

Based on a German Credit dataset with customer attributes such as:

* Age
* Job level
* Housing type
* Saving & checking accounts
* Credit amount
* Loan duration
* Purpose of loan

Custom domain features were engineered to improve model performance.

---

#  Feature Engineering (Core Strength of Project)

Created intelligent financial indicators:

* Credit_per_month
* Saving_score
* Checking_score
* Financial_strength
* Stress_score
* Business_risk
* High_risk_combo
* Both_Accounts_Null

These simulate real risk signals used by banks.

---

#  Model Details

Algorithm: XGBoost (Boosting)

Why Boosting?

* Handles nonlinear patterns
* Strong performance on tabular data
* Robust to feature interactions
* Industry-preferred for risk scoring

Metrics achieved:

* Accuracy ~72–75%
* ROC-AUC ~0.75+
* Balanced performance for detecting high-risk customers

---

#  Explainability (SHAP)

The system explains predictions by highlighting:

* Top risk-driving features
* Feature impact per prediction
* Transparent decision support

This is critical in regulated financial environments.

---

#  FastAPI Backend

The API handles:

* Raw input intake
* Feature transformation
* Model prediction
* Risk scoring
* SHAP explanation

Endpoint:

POST /predict

Input:
Raw customer details

Output:

* Default probability
* Risk score
* Decision
* Top risk factors

---

#  Streamlit Frontend

Interactive UI with:

* Numeric inputs (text boxes)
* Categorical inputs (dropdowns)
* Risk meter gauge visualization
* Color-coded decision box
* Top risk factors display

This simulates a real fintech dashboard.

---

#  Installation & Setup

## 1) Clone Repository

```
git clone <your-repo-link>
cd credit-risk-system
```

## 2) Install Dependencies

```
pip install -r requirements.txt
```

Or manually:

```
pip install pandas numpy scikit-learn xgboost shap fastapi uvicorn streamlit plotly requests joblib
```

---

#  Run the Project

## Start FastAPI Backend

```
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## Start Streamlit Frontend

Open new terminal:

```
streamlit run app/streamlit_app.py
```

---

#  Example Output

* Default Probability: 0.49
* Risk Score: 50.6
* Decision: Review
* Top Risk Factors:

  * Credit amount
  * Saving score
  * Checking score


---


#  Author

Aryan Yadav
projects.aky@gmail.com
---



