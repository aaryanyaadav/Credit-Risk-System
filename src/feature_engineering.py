import pandas as pd

Median_credit_amount=130.33

def tranform_input(data_dict):

    df=pd.DataFrame([data_dict])

    # for credit per month
    df["Credit_per_month"] = df["Credit amount"] / df["Duration"]

    # saving score mapping
    saving_map = {
        "little": 1,
        "moderate": 2,
        "rich": 3,
        "quite rich": 4
    }
    df["Saving_score"] = df["Saving accounts"].map(saving_map).fillna(0)

    # Checking score mapping
    checking_map = {
        "little": 1,
        "moderate": 2,
        "rich": 3
    }
    df["Checking_score"] = df["Checking account"].map(checking_map).fillna(0)

    # Financial strength
    df["Financial_strength"] = df["Saving_score"] + df["Checking_score"]

    # Stress score
    df["Stress_score"] = (
        (df["Saving accounts"].isnull()).astype(int) +
        (df["Checking account"].isnull()).astype(int) +
        (df["Housing"] == "rent").astype(int)
    )

    # Business risk
    df["Business_risk"] = (
        (df["Purpose"] == "business") &
        (df["Saving accounts"].isnull()) &
        (df["Checking account"].isnull())
    ).astype(int)

    # High risk combo
    df["High_risk_combo"] = (
        (df["Stress_score"] >= 2) &
        (df["Credit_per_month"] > Median_credit_amount)
    ).astype(int)


    # Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Housing
    df["Housing"] = df["Housing"].map({
        "own": 2,
        "rent": 1,
        "free": 0
    })

    # One-hot encode Purpose
    df = pd.get_dummies(df, columns=["Purpose"])

    purpose_cols = [
        "Purpose_radio/TV",
        "Purpose_education",
        "Purpose_furniture/equipment",
        "Purpose_car",
        "Purpose_business",
        "Purpose_domestic appliances",
        "Purpose_repairs",
        "Purpose_vacation/others"
    ]

    # Ensure all columns exist
    for col in purpose_cols:
        if col not in df.columns:
            df[col] = 0

    # Drop raw columns not used by model
    df = df.drop([
        "Saving accounts",
        "Checking account"
    ], axis=1)

    return df