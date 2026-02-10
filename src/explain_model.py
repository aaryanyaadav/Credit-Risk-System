import pandas as pd
import joblib
import shap

df=pd.read_csv("data/processed_data.csv")

X=df.drop("Risk",axis=1)

model=joblib.load("models/xgboost_model.pkl")

# explainer for trained model
explainer=shap.TreeExplainer(model)

#explination for the test data predictions 
shap_values=explainer.shap_values(X)

shap.summary_plot(shap_values,X)
