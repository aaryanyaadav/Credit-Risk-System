import pandas as pd

# Median used during training
Median_credit_amount = 130.33

# EXACT column order used during training (VERY IMPORTANT)
MODEL_COLUMNS = [
    'Age',
    'Sex',
    'Job',
    'Housing',
    'Credit amount',
    'Duration',
    'Both_Accounts_Null',
    'Credit_per_month',
    'Stress_score',
    'Saving_score',
    'Checking_score',
    'Financial_strength',
    'Business_risk',
    'High_risk_combo',
    'Purpose_car',
    'Purpose_domestic appliances',
    'Purpose_education',
    'Purpose_furniture/equipment',
    'Purpose_radio/TV',
    'Purpose_repairs',
    'Purpose_vacation/others'
]


def transform_input(data_dict):

    # Convert user input to dataframe
    df = pd.DataFrame([data_dict])



    # Credit per month
    df["Credit_per_month"] = df["Credit amount"] / df["Duration"]

    # Saving score mapping
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

    # Both accounts null
    df["Both_Accounts_Null"] = (
        df["Saving accounts"].isnull() &
        df["Checking account"].isnull()
    ).astype(int)


    # Sex encoding
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Housing encoding
    df["Housing"] = df["Housing"].map({
        "own": 2,
        "rent": 1,
        "free": 0
    })

    # One-hot encode Purpose
    df = pd.get_dummies(df, columns=["Purpose"])

    # Only the columns used during training
    purpose_cols = [
        "Purpose_car",
        "Purpose_domestic appliances",
        "Purpose_education",
        "Purpose_furniture/equipment",
        "Purpose_radio/TV",
        "Purpose_repairs",
        "Purpose_vacation/others"
    ]

    # Ensure all required columns exist
    for col in purpose_cols:
        if col not in df.columns:
            df[col] = 0


    # Droping the columns

    df = df.drop([
        "Saving accounts",
        "Checking account"
    ], axis=1)


    df = df[MODEL_COLUMNS]

    return df
