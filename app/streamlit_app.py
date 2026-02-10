import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("Credit Risk Scoring System")
st.write("Enter the detailsof the applicant")

age=st.number_input("Age",min_value=10,value=30)
job=st.selectbox("Job Level",[0,1,2,3])
credit_amount=st.number_input("Credit Amount",min_value=1,value=5000)
duration=st.number_input("Loan Duration",min_value=1,value=24)
sex = st.selectbox("Sex", ["male", "female"])

housing = st.selectbox(
    "Housing Type",
    ["own", "rent", "free"]
)

saving_accounts = st.selectbox(
    "Saving Accounts",
    ["little", "moderate", "rich", "quite rich", None]
)

checking_account = st.selectbox(
    "Checking Account",
    ["little", "moderate", "rich", None]
)

purpose = st.selectbox(
    "Loan Purpose",
    [
        "radio/TV",
        "education",
        "furniture/equipment",
        "car",
        "business",
        "domestic appliances",
        "repairs",
        "vacation/others"
    ]
)
if st.button("Predict Risk"):

    input_data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving_accounts,
        "Checking account": checking_account,
        "Credit amount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=input_data
        )

        result = response.json()

        st.subheader(" Prediction Result")

        st.write(f"**Default Probability:** {result['default_probability']}")
        st.write(f"**Risk Score:** {result['risk_score']}")
        st.write(f"**Decision:** {result['decision']}")

        st.subheader(" Top Risk Factors")
        for factor in result["top_risk_factors"]:
            st.write(f"- {factor}")

    except:
        st.error(" Could not connect to API. Is FastAPI running?")