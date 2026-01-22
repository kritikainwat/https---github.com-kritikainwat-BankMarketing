import streamlit as st
import joblib
import pandas as pd

# Load trained model pipeline
model = joblib.load("C:/Users/kulde/OneDrive/Desktop/BankMarketingProject/models/best_rf_pipeline.joblib")


st.title("📊 Bank Term Deposit Prediction App")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ["admin.","blue-collar","entrepreneur","housemaid","management",
                           "retired","self-employed","services","student","technician","unemployed","unknown"])
marital = st.selectbox("Marital", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Default Credit?", ["yes", "no"])
balance = st.number_input("Balance", value=500)
housing = st.selectbox("Housing Loan?", ["yes", "no"])
loan = st.selectbox("Personal Loan?", ["yes", "no"])
contact = st.selectbox("Contact Communication", ["cellular", "telephone", "unknown"])
day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
duration = st.number_input("Call Duration (seconds)", value=200)
campaign = st.number_input("Number of Campaign Contacts", value=1)
pdays = st.number_input("Days Since Last Contact (-1 if never)", value=-1)
previous = st.number_input("Previous Contacts", value=0)
poutcome = st.selectbox("Previous Outcome", ["success","failure","other","unknown"])

# DataFrame
input_dict = {
    "age": [age], "job": [job], "marital": [marital], "education": [education],
    "default": [default], "balance": [balance], "housing": [housing], "loan": [loan],
    "contact": [contact], "day": [day], "month": [month], "duration": [duration],
    "campaign": [campaign], "pdays": [pdays], "previous": [previous], "poutcome": [poutcome]
}
input_df = pd.DataFrame(input_dict)

# 🔹 Add engineered features here
import numpy as np
input_df["balance_clipped"] = input_df["balance"].clip(upper=2500)
input_df["balance_log"] = np.log1p(input_df["balance"])
input_df["pdays_imputed"] = input_df["pdays"].replace({-1: pd.NA}).fillna(999)
input_df["contacted_before"] = (input_df["pdays"] != -1).astype(int)

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if pred == 1:
        st.success(f"✅ Likely to Subscribe (Confidence: {proba:.2f})")
    else:
        st.error(f"❌ Not Likely to Subscribe (Confidence: {proba:.2f})")
