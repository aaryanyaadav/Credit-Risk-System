import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report
import joblib

df = pd.read_csv("data/processed_data.csv")

#test-train split

x=df.drop("Risk",axis=1)
y=df["Risk"]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

model = XGBClassifier(

    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])

)

model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("model accuracy:",accuracy_score(Y_test,y_pred))
print("model confusion matrix :",confusion_matrix(Y_test,y_pred))
print("ROC-AUC:", roc_auc_score(Y_test, y_prob))

print(classification_report(Y_test, y_pred))

joblib.dump(model, "models/xgboost_model.pkl")

print("\nModel saved")