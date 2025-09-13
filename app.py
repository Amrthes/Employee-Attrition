import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load model, scaler, and columns
# -------------------------------
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.title("Employee Attrition Prediction")
st.write("Enter employee details to predict attrition probability.")

# -------------------------------
# User Inputs
# -------------------------------
Age = st.number_input("Age", 18, 60, 30)
BusinessTravel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
Department = st.selectbox("Department", ["Human Resources", "Research & Development", "Sales"])
DistanceFromHome = st.number_input("Distance From Home", 0, 50, 5)
Education = st.selectbox("Education Level", [1,2,3,4,5])
EducationField = st.selectbox("Education Field", ["Life Sciences","Other","Medical","Marketing","Technical Degree","Human Resources"])
EnvironmentSatisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4])
Gender = st.selectbox("Gender", ["Male","Female"])
JobRole = st.selectbox("Job Role", ["Sales Executive","Research Scientist","Laboratory Technician",
                                   "Manufacturing Director","Healthcare Representative","Manager",
                                   "Sales Representative","Research Director","Human Resources"])
JobSatisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])
MaritalStatus = st.selectbox("Marital Status", ["Single","Married","Divorced"])
MonthlyIncome = st.number_input("Monthly Income", 1000, 50000, 5000)
OverTime = st.selectbox("OverTime", ["No","Yes"])

# -------------------------------
# Prepare input DataFrame
# -------------------------------
input_dict = {
    "Age": Age,
    "BusinessTravel": BusinessTravel,
    "Department": Department,
    "DistanceFromHome": DistanceFromHome,
    "Education": Education,
    "EducationField": EducationField,
    "EnvironmentSatisfaction": EnvironmentSatisfaction,
    "Gender": Gender,
    "JobRole": JobRole,
    "JobSatisfaction": JobSatisfaction,
    "MaritalStatus": MaritalStatus,
    "MonthlyIncome": MonthlyIncome,
    "OverTime": OverTime
}

input_df = pd.DataFrame([input_dict])

# -------------------------------
# Encode categorical features
# -------------------------------
categorical_mappings = {
    "BusinessTravel":{"Non-Travel":0,"Travel_Rarely":1,"Travel_Frequently":2},
    "Department":{"Human Resources":0,"Research & Development":1,"Sales":2},
    "EducationField":{"Life Sciences":0,"Other":1,"Medical":2,"Marketing":3,"Technical Degree":4,"Human Resources":5},
    "Gender":{"Male":0,"Female":1},
    "JobRole":{"Sales Executive":0,"Research Scientist":1,"Laboratory Technician":2,"Manufacturing Director":3,
               "Healthcare Representative":4,"Manager":5,"Sales Representative":6,"Research Director":7,"Human Resources":8},
    "MaritalStatus":{"Single":0,"Married":1,"Divorced":2},
    "OverTime":{"No":0,"Yes":1}
}

for col, mapping in categorical_mappings.items():
    input_df[col] = input_df[col].map(mapping)

# -------------------------------
# Add missing columns with default 0
# -------------------------------
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[model_columns]

# Scale numeric features
input_scaled = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Attrition"):
    prob = model.predict_proba(input_scaled)[:,1][0]
    pred = model.predict(input_scaled)[0]
    
    st.write(f"**Attrition Probability:** {prob:.2f}")
    if pred == 1:
        st.warning("⚠️ This employee is likely to leave!")
    else:
        st.success("✅ This employee is likely to stay.")
