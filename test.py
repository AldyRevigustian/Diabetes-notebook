import joblib
import pandas as pd
import numpy as np

# Load scaler dan model
scaler = joblib.load("Models/scaler.joblib")
model = joblib.load("Models/CatBoost.pkl")

# Kolom yang digunakan saat training (harus sesuai!)
columns = [
    "Age","Gender","BMI","SBP","DBP","FPG","Chol","Tri","HDL","LDL","ALT","AST","BUN","CCR","FFPG","smoking","drinking","family_history"
]

# Input baru (pastikan urutan sesuai kolom)
X_new = pd.DataFrame([[40,1,28.90,119.00,72.00,5.50,4.61,0.79,1.22,2.88,92.10,64.80,6.96,81.10,4.80,3,2,0]], columns=columns)

# Transformasi dengan scaler (ColumnTransformer)
X_new_scaled = scaler.transform(X_new)

# Prediksi
y_pred = model.predict(X_new_scaled)

# Jika model mendukung probabilitas
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_new_scaled)
    print("Probabilitas Prediksi:", y_proba)

print("Hasil Prediksi:", y_pred)
