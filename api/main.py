from fastapi import FastAPI
import pandas as pd
import shap
from xgboost import XGBClassifier

from src.feature_engineering import transform_input

app = FastAPI(title="Credit risk prediction api")

#Load model using XGBoost native format
model = XGBClassifier()
model.load_model("models/xgboost_model.json")

#SHAP explainer
explainer = shap.TreeExplainer(model)

def decision(score):
    if score >= 70:
        return "Approve"
    elif score >= 40:
        return "Review"
    else:
        return "Reject"

#Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "credit-risk-api",
        "uptime": "running",
        "version": "1.0.0"
    }

@app.post("/predict")
def predict(data: dict):
    process_data_df = transform_input(data)

    prob = model.predict_proba(process_data_df)[0][1]
    risk_score = (1 - prob) * 100
    final_decision = decision(risk_score)

    shap_values = explainer.shap_values(process_data_df)[0]

    features_impacts = pd.DataFrame({
        "feature": process_data_df.columns,
        "impact": shap_values
    })

    top_risk = (
        features_impacts
        .sort_values(by="impact", ascending=False)
        .head(3)["feature"]
        .tolist()
    )

    return {
        "default_probability": round(float(prob), 3),
        "risk_score": round(float(risk_score), 2),
        "decision": final_decision,
        "top_risk_factors": top_risk
    }