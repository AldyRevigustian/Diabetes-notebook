import streamlit as st

st.set_page_config(page_title="🧾 Dataset Glossary", layout="wide")

st.title("🧾 Dataset Glossary")
st.markdown(
    "This page provides definitions and units for all input features used in the diabetes prediction app."
)
st.markdown("---")

# === Glossary Entries ===

st.markdown("### Age")
st.write(
    "The age of the patient in years at the time of data collection (measured in **years**)."
)
st.markdown("---")

st.markdown("### Gender")
st.write("Biological sex of the patient:")
st.markdown("- **1:** Male")
st.markdown("- **2:** Female")
st.markdown("---")

st.markdown("### BMI (Body Mass Index)")
st.write(
    "Body Mass Index calculated as weight divided by height squared (measured in **kg/m²**). Used to classify underweight, normal weight, overweight, or obesity."
)
st.markdown("---")

st.markdown("### SBP (Systolic Blood Pressure)")
st.write("The pressure in arteries when the heart beats (measured in **mmHg**).")
st.markdown("---")

st.markdown("### DBP (Diastolic Blood Pressure)")
st.write(
    "The pressure in arteries when the heart rests between beats (measured in **mmHg**)."
)
st.markdown("---")

st.markdown("### FPG (Fasting Plasma Glucose)")
st.write(
    "Blood glucose level measured after fasting for at least 8 hours (measured in **mmol/L**)."
)
st.markdown("---")

st.markdown("### FFPG (Final Visit Fasting Plasma Glucose)")
st.write(
    "Final Fasting Plasma Glucose measured at the last clinical visit (measured in **mmol/L**)."
)
st.markdown("---")

st.markdown("### Chol (Total Cholesterol)")
st.write("Total amount of cholesterol in the blood (measured in **mmol/L**).")
st.markdown("---")

st.markdown("### Tri (Triglycerides)")
st.write(
    "Type of fat (lipid) in the blood. High levels may increase the risk of heart disease (measured in **mmol/L**)."
)
st.markdown("---")

st.markdown("### HDL (High-Density Lipoprotein)")
st.write(
    "Good cholesterol that helps remove other forms of cholesterol from the bloodstream (measured in **mmol/L**)."
)
st.markdown("---")

st.markdown("### LDL (Low-Density Lipoprotein)")
st.write(
    "Bad cholesterol that can build up and block blood vessels (measured in **mmol/L**)."
)
st.markdown("---")

st.markdown("### ALT (Alanine Aminotransferase)")
st.write(
    "Liver enzyme. High levels may indicate liver damage or disease (measured in **U/L**)."
)
st.markdown("---")

st.markdown("### AST (Aspartate Aminotransferase)")
st.write(
    "Enzyme found in liver and other tissues. High levels may also indicate liver or muscle damage (measured in **U/L**)."
)
st.markdown("---")

st.markdown("### BUN (Blood Urea Nitrogen)")
st.write(
    "Amount of nitrogen in the blood that comes from urea. Used to evaluate kidney function (measured in **mmol/L**)."
)
st.markdown("---")

st.markdown("### CCR (Creatinine Clearance Rate)")
st.write(
    "Estimates how well the kidneys filter creatinine. A marker of kidney function (measured in **µmol/L**)."
)
st.markdown("---")

st.markdown("### Smoking")
st.write("Patient's smoking status:")
st.markdown("- **1:** Never Smoked")
st.markdown("- **2:** Former Smoker")
st.markdown("- **3:** Current Smoker")
st.markdown("---")

st.markdown("### Drinking")
st.write("Patient's alcohol consumption level:")
st.markdown("- **1:** Level 1 (Low or Never)")
st.markdown("- **2:** Level 2 (Moderate)")
st.markdown("- **3:** Level 3 (High or Current)")
st.markdown("---")

st.markdown("### Family History")
st.write("Indicates if the patient has a family history of diabetes:")
st.markdown("- **0:** No")
st.markdown("- **1:** Yes")
st.markdown("---")

st.markdown("### Diabetes (Target Variable)")
st.write("Indicates whether the patient is diabetic:")
st.markdown("- **0:** Non-Diabetic")
st.markdown("- **1:** Diabetic")
st.markdown("---")
