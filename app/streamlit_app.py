import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")

st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        color:#1f77b4;
    }
    .card {
        padding:20px;
        border-radius:15px;
        background-color:#111827;
        box-shadow:0px 4px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# header
st.markdown("<div class='main-title'> Credit Risk Scoring Dashboard</div>", unsafe_allow_html=True)
st.write("AI-powered system to predict loan default risk")

st.divider()

# sidebar
st.sidebar.header("  Applicant Information")

age = st.sidebar.number_input("Age", 18, 100, 30)
job = st.sidebar.selectbox("Job Level", [0, 1, 2, 3])
credit_amount = st.sidebar.number_input("Credit Amount", min_value=0, value=5000)
duration = st.sidebar.number_input("Loan Duration (months)", min_value=1, value=24)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
housing = st.sidebar.selectbox("Housing Type", ["own", "rent", "free"])

saving_accounts = st.sidebar.selectbox(
    "Saving Accounts",
    ["little", "moderate", "rich", "quite rich", None]
)

checking_account = st.sidebar.selectbox(
    "Checking Account",
    ["little", "moderate", "rich", None]
)

purpose = st.sidebar.selectbox(
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

predict_btn = st.sidebar.button(" Predict Risk")

#prediction
if predict_btn:

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

        risk_score = result["risk_score"]
        decision = result["decision"]
        default_prob = result["default_probability"]
        top_factors = result["top_risk_factors"]

        st.subheader(" Prediction Results")

        # applicant summary
        st.subheader(" Applicant Summary")

        with st.container():
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("#### Personal Info")
                st.write(f"**Age:** {age}")
                st.write(f"**Sex:** {sex}")
                st.write(f"**Job Level:** {job}")

            with c2:
                st.markdown("#### Financial Info")
                st.write(f"**Credit Amount:** ₹{credit_amount:,}")
                st.write(f"**Loan Duration:** {duration} months")
                st.write(f"**Housing:** {housing}")

            with c3:
                st.markdown("#### Account Details")
                st.write(f"**Saving Account:** {saving_accounts}")
                st.write(f"**Checking Account:** {checking_account}")
                st.write(f"**Purpose:** {purpose}")

        st.divider()


        col1, col2, col3 = st.columns(3)

        # metirc card
        col1.metric("Approval Score", f"{risk_score:.2f}")
        col2.metric("Default Probability", f"{default_prob:.2f}")
        col3.metric("Decision", decision)

        st.divider()

        # guage
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Approval Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # decision box
        if decision == "Approve":
            st.success(" Loan Approved")
        elif decision == "Review":
            st.warning(" Needs Manual Review")
        else:
            st.error(" High Risk –  Reject")

        #
        st.subheader("Default Risk Level")
        st.progress(float(default_prob))

        #factores affecting the decision
        st.subheader(" Top Factors Causing the Decison")

        for factor in top_factors:
            st.markdown(f"- {factor}")

    except:
        st.error("Server went to have a small coffeex")
