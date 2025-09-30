import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load model and features
model = joblib.load("random_forest_model.pkl")
feature_names = joblib.load("features.pkl")

# -------------------- UI Layout --------------------
st.set_page_config(page_title="TB / NCD Risk Prediction", page_icon="🩺", layout="wide")

st.title("🩺 TB / NCD Risk Prediction App")
st.markdown("""
Welcome to the **TB / NCD Risk Prediction Tool** 🎯  
This app helps clinicians and researchers predict the risk of **Tuberculosis (TB)** or **Non-Communicable Diseases (NCDs)** among patients on ART.  

👉 Adjust patient details in the sidebar, and click **Predict** to see results.
""")

st.image("https://img.freepik.com/free-vector/medical-background-design_1212-105.jpg", use_column_width=True)

# -------------------- Sidebar Input --------------------
st.sidebar.header("📋 Patient Information")

input_data = {}
for col in feature_names:
    if col == "Sex":
        input_data[col] = st.sidebar.selectbox("Sex", ["Male", "Female"])
    elif col == "Population Type":
        input_data[col] = st.sidebar.selectbox("Population Type", ["General Population", "Key Population"])
    elif "age" in col.lower():
        input_data[col] = st.sidebar.slider("Age", 0, 100, 30)
    elif "bmi" in col.lower():
        input_data[col] = st.sidebar.slider("BMI", 10, 50, 22)
    elif "bp_sys" in col.lower():
        input_data[col] = st.sidebar.slider("Systolic BP", 80, 200, 120)
    elif "bp_dia" in col.lower():
        input_data[col] = st.sidebar.slider("Diastolic BP", 50, 120, 80)
    elif "vl" in col.lower():
        input_data[col] = st.sidebar.number_input(f"{col}", min_value=0, step=1)
    else:
        input_data[col] = st.sidebar.number_input(f"{col}", min_value=0.0, step=1.0)

# Convert input to dataframe
df_input = pd.DataFrame([input_data])

# Ensure correct column order
df_input = df_input.reindex(columns=feature_names)

# -------------------- Prediction --------------------
if st.sidebar.button("Predict"):
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Prediction")
        st.write("**Positive Risk** ⚠️" if pred == 1 else "**Negative Risk** ✅")
        st.write(f"**Probability of risk:** {prob:.2f}")

    with col2:
        st.subheader("📊 Probability Gauge")
        fig, ax = plt.subplots()
        ax.bar(["Negative", "Positive"], [1 - prob, prob], color=["green", "red"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    # -------------------- Feature Importance --------------------
    st.subheader("🔎 Feature Importance")
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[-10:]  # top 10
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx], color="skyblue")
        ax.set_title("Top 10 Features Driving Prediction")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")

st.markdown("---")
st.markdown("👨‍⚕️ *Developed for ART program data analysis and prediction*")
