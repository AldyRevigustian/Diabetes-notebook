# pages/3_App.py

import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="🩺 Diabetes Prediction App")

st.title("🩺 Diabetes Prediction App")
st.markdown(
    """
Enter patient details below to get predictions from various machine learning models.
Hover over the (?) icons next to each field for more information.
"""
)


# --- Model Loading ---
@st.cache_resource
def load_model(relative_path):
    try:
        script_dir = os.path.dirname(__file__)
        if not script_dir:
            script_dir = "."
        abs_path = os.path.abspath(os.path.join(script_dir, relative_path))
        if os.path.exists(abs_path):
            return joblib.load(abs_path)
        else:
            st.error(f"File not found: {abs_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# --- Load Models ---
model_paths = {
    "XGBoost": "../Models/XGBoost.pkl",
    "Random Forest": "../Models/Random Forest.pkl",
    "Gradient Boosting": "../Models/Gradient Boosting.pkl",
    "Decision Tree": "../Models/Decision Tree.pkl",
    "CatBoost": "../Models/CatBoost.pkl",
}
# --- Load Models ---
models = {}
all_models_loaded = True

with st.sidebar:
    st.header("Model Status")
    failed_models = []

    for name, path in model_paths.items():
        model = load_model(path)
        if model:
            models[name] = model
        else:
            all_models_loaded = False
            failed_models.append(name)

    if failed_models:
        for name in failed_models:
            st.error(f"{name} failed to load")
    else:
        st.success("✅ All models loaded successfully!")

# --- Load Scaler ---
scaler = load_model("../Models/scaler.joblib")

# --- Feature Order ---
features = [
    "Age",
    "Gender",
    "BMI",
    "SBP",
    "DBP",
    "FPG",
    "Chol",
    "Tri",
    "HDL",
    "LDL",
    "ALT",
    "AST",
    "BUN",
    "CCR",
    "FFPG",
    "smoking",
    "drinking",
    "family_history",
]
# Default values berdasarkan data yang diberikan
default_values = {
    "Age": 73,
    "Gender": 2,
    "BMI": 23.90,
    "SBP": 169.00,
    "DBP": 74.00,
    "FPG": 5.02,
    "Chol": 5.34,
    "Tri": 1.30,
    "HDL": 1.28,
    "LDL": 2.91,
    "ALT": 14.20,
    "AST": 26.60,
    "BUN": 8.77,
    "CCR": 113.20,
    "FFPG": 7.00,
    "smoking": 3,
    "drinking": 3,
    "family_history": 0,
}

input_data = {}

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], index=1)
    input_data["Gender"] = 1 if gender == "Male" else 2
    input_data["Age"] = st.number_input(
        "Age (years old)", 0, 120, default_values["Age"]
    )
    input_data["BMI"] = st.number_input("BMI kg/m²", 10.0, 60.0, default_values["BMI"])
    fam_hist = st.selectbox("Family History of Diabetes", ["No", "Yes"], index=0)
    input_data["family_history"] = 0 if fam_hist == "No" else 1
    smoking = st.selectbox(
        "Smoking Status", ["Never Smoked", "Former Smoker", "Current Smoker"], index=2
    )
    input_data["smoking"] = {
        "Never Smoked": 1,
        "Former Smoker": 2,
        "Current Smoker": 3,
    }[smoking]
    drinking = st.selectbox(
        "Alcohol Consumption Level",
        ["Level 1 (Low or Never)", "Level 2 (Moderate)", "Level 3 (High or Current)"],
        index=2,
    )
    input_data["drinking"] = {
        "Level 1 (Low or Never)": 1,
        "Level 2 (Moderate)": 2,
        "Level 3 (High or Current)": 3,
    }[drinking]

with col2:
    input_data["Chol"] = st.number_input(
        "Total Cholesterol (mmol/L)", 1.0, 15.0, default_values["Chol"]
    )
    input_data["HDL"] = st.number_input(
        "High-Density Lipoprotein (HDL) Cholesterol (mmol/L)",
        0.1,
        4.0,
        default_values["HDL"],
    )
    input_data["LDL"] = st.number_input(
        "Low-Density Lipoprotein (LDL) Cholesterol (mmol/L)",
        0.1,
        10.0,
        default_values["LDL"],
    )
    input_data["FPG"] = st.number_input(
        "Fasting Plasma Glucose (FPG) (mmol/L)", 1.0, 20.0, default_values["FPG"]
    )
    input_data["FFPG"] = st.number_input(
        "Final Visit Fasting Plasma Glucose (FFPG) (mmol/L)",
        1.0,
        30.0,
        default_values["FFPG"],
    )
    input_data["BUN"] = st.number_input(
        "Blood Urea Nitrogen (BUN) (mmol/L)", 1.0, 20.0, default_values["BUN"]
    )

with col3:
    input_data["ALT"] = st.number_input(
        "Alanine Aminotransferase (ALT) (U/L)", 0.0, 300.0, default_values["ALT"]
    )
    input_data["AST"] = st.number_input(
        "Aspartate Aminotransferase (AST) (U/L)", 0.0, 300.0, default_values["AST"]
    )
    input_data["SBP"] = st.number_input(
        "Systolic Blood Pressure (SBP) (mmHg)", 50.0, 250.0, default_values["SBP"]
    )
    input_data["DBP"] = st.number_input(
        "Diastolic Blood Pressure (DBP) (mmHg)", 30.0, 150.0, default_values["DBP"]
    )
    input_data["Tri"] = st.number_input(
        "Triglycerides Level (mmol/L)", 0.1, 15.0, default_values["Tri"]
    )
    input_data["CCR"] = st.number_input(
        "Creatinine Clearance Rate (CCR) (µmol/L)", 10.0, 300.0, default_values["CCR"]
    )


st.write("---")

# --- Prediction ---
if st.button("Predict Diabetes Status"):
    if not all_models_loaded:
        st.error("Some models failed to load. Cannot proceed.")
    elif scaler is None:
        st.error("Scaler failed to load. Cannot proceed.")
    else:
        try:
            input_df = pd.DataFrame([input_data])[features]
            st.subheader("Input Data")
            st.dataframe(input_df)

            input_scaled = scaler.transform(input_df)
            st.subheader("Model Predictions")
            results = {}

            for model_name, model in models.items():
                try:
                    pred = model.predict(input_scaled)[0]
                    prob = (
                        model.predict_proba(input_scaled)[0][1]
                        if hasattr(model, "predict_proba")
                        else None
                    )
                    label = "Diabetic" if pred == 1 else "Non-Diabetic"
                    results[model_name] = (label, prob)
                except Exception as e:
                    st.error(f"Prediction error for {model_name}: {e}")
                    results[model_name] = ("Error", None)

            cols = st.columns(len(results))
            for i, (name, (label, prob)) in enumerate(results.items()):
                with cols[i]:
                    st.metric(name, label)
                    if prob is not None:
                        st.write(f"Probability: {prob:.2f}")
                    else:
                        st.write("No probability")

        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    ()
