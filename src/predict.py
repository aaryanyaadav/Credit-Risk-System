import pandas as pd
import joblib
import shap
model=joblib.load("models/xgboost_model.pkl")

df=pd.read_csv("data/processed_data.csv")

X=df.drop("Risk",axis=1)

# picking a random customer
sample = X.iloc[[1]]

# predict the probabillity with the help of the trained model
default_prob = model.predict_proba(sample)[0][1]   #probability of 0 means  good & 1 means bad (default), so the more probability one is choosen

# we make a risk score
risk_score = (1 - default_prob) * 100

# decision logic that we are using
def decision(score):
    if score >= 70:
        return "Approve"
    elif score >= 40:
        return "Review"
    else:
        return "Reject"

# now shap to add give the features that are causing the specific prediction

# explainer for trained model
explainer=shap.TreeExplainer(model)

#explination for the test data predictions 
shap_values=explainer.shap_values(sample)


# Get feature contributions
shap_contrib = pd.DataFrame({
    "feature": X.columns,
    "impact": shap_values[0]
})

# Sort by strongest impact toward default (positive side)
top_risk = shap_contrib.sort_values(by="impact", ascending=False).head(3)


print("Default Probability:", round(default_prob, 3))
print("Risk Score:", round(risk_score, 2))
print("Decision:", decision(risk_score))

print("\nTop risk factors:")
for f in top_risk["feature"]:
    print(f)

