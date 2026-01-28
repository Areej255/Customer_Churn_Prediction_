import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model = joblib.load(os.path.join(BASE_DIR, "churn_prediction_model.pkl"))
# scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
# feature_names = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))


# Load trained objects
model = joblib.load("churn_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Customer Churn Dashboard", layout="centered")

st.title("📉 Customer Churn Prediction Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload Customer Dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Drop ID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges if exists
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # same encoding as training
    df = pd.get_dummies(df)
    
    # Align columns with training
    df = df.reindex(columns=feature_names, fill_value=0)
    
    x = df
    # Scale
    x_scaled = scaler.transform(x)

    # Predict
    churn_prob = model.predict_proba(x_scaled)[:,1]
    df["Churn_Probability"] = churn_prob
    df["Risk_Level"] = df["Churn_Probability"].apply(lambda x: "High" if x>0.7 else "Low")

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("High Risk Customers", (df["Risk_Level"]=="High").sum())
    col3.metric("Avg Churn Probability", round(df["Churn_Probability"].mean(),2))

    st.subheader("High Risk Customers")
    st.dataframe(df[df["Risk_Level"]=="High"].sort_values("Churn_Probability", ascending=False))

    # Feature Importance
    st.subheader("Why Customers Churn")
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Impact": model.coef_[0]
    }).sort_values(by="Impact", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance["Feature"][:10], importance["Impact"][:10])
    ax.invert_yaxis()
    st.pyplot(fig)
