import streamlit as st
import pandas as pd
import joblib
import os
import sys


st.set_page_config(layout="wide", page_title="🩺 Aplikasi Prediksi Diabetes")

st.title("🩺 Aplikasi Prediksi Diabetes")
st.markdown(
    """
Masukkan detail pasien untuk memprediksi dari berbagai model machine learning.
Arahkan kursor ke ikon (?) untuk melihat penjelasan singkat.
"""
)


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
            st.error(f"Berkas tidak ditemukan: {abs_path}")
            return None
    except Exception as e:
        st.error(f"Kesalahan saat memuat model: {e}")
        return None


model_paths = {
    "XGBoost": "../Models/XGBoost.pkl",
    "Random Forest": "../Models/Random Forest.pkl",
    "Gradient Boosting": "../Models/Gradient Boosting.pkl",
    "Decision Tree": "../Models/Decision Tree.pkl",
    "CatBoost": "../Models/CatBoost.pkl",
}
models = {}
all_models_loaded = True

with st.sidebar:
    st.header("Status Model")
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
            st.error(f"{name} gagal dimuat")
    else:
        st.success("✅ Semua model berhasil dimuat!")


scaler = load_model("../Models/scaler.joblib")


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
# # Non Diabetes
default_values = {
    "Age": 35,
    "Gender": 2,
    "BMI": 21.20,
    "SBP": 98.00,
    "DBP": 63.00,
    "FPG": 5.30,
    "Chol": 3.90,
    "Tri": 1.00,
    "HDL": 1.36,
    "LDL": 2.07,
    "ALT": 33.00,
    "AST": 29.00,
    "BUN": 4.23,
    "CCR": 68.00,
    "FFPG": 4.70,
    "smoking": 3,
    "drinking": 3,
    "family_history": 0,
}

# default_values = {
#     "Age": 73,
#     "Gender": 2,
#     "BMI": 23.90,
#     "SBP": 169.00,
#     "DBP": 74.00,
#     "FPG": 5.02,
#     "Chol": 5.34,
#     "Tri": 1.30,
#     "HDL": 1.28,
#     "LDL": 2.91,
#     "ALT": 14.20,
#     "AST": 26.60,
#     "BUN": 8.77,
#     "CCR": 113.20,
#     "FFPG": 7.00,
#     "smoking": 3,
#     "drinking": 3,
#     "family_history": 0,
# }

input_data = {}


col1, col2, col3 = st.columns(3)
with col1:

    gender_options = ["Pria", "Wanita"]
    gender_default_index = gender_options.index("Wanita")
    gender = st.selectbox(
        "Jenis Kelamin",
        gender_options,
        index=gender_default_index,
        help="Pilih jenis kelamin biologis pasien.",
    )
    input_data["Gender"] = 1 if gender == "Pria" else 2

    input_data["Age"] = st.number_input(
        "Usia (tahun)",
        min_value=0,
        max_value=120,
        value=default_values["Age"],
        help="Masukkan usia pasien dalam satuan tahun.",
    )

    input_data["BMI"] = st.number_input(
        "IMT (kg/m²)",
        min_value=10.0,
        max_value=60.0,
        value=default_values["BMI"],
        step=0.1,
        help="Indeks Massa Tubuh, dihitung sebagai berat(kg) / tinggi(m)^2.",
    )

    fam_hist_options = ["Tidak", "Ya"]
    fam_hist_default_index = fam_hist_options.index("Tidak")
    fam_hist = st.selectbox(
        "Riwayat Diabetes Keluarga",
        fam_hist_options,
        index=fam_hist_default_index,
        help="Apakah pasien memiliki anggota keluarga dengan riwayat diabetes?",
    )
    input_data["family_history"] = 0 if fam_hist == "Tidak" else 1

    smoking_options = ["Tidak Pernah Merokok", "Pernah Merokok", "Perokok Aktif"]
    smoking_map = {
        "Tidak Pernah Merokok": 3,
        "Pernah Merokok": 2,
        "Perokok Aktif": 1,
    }
    smoking_default_value_text = [
        k for k, v in smoking_map.items() if v == default_values["smoking"]
    ][0]
    smoking_default_index = smoking_options.index(smoking_default_value_text)
    smoking = st.selectbox(
        "Status Merokok",
        smoking_options,
        index=smoking_default_index,
        help="Pilih status merokok pasien saat ini atau di masa lalu.",
    )
    input_data["smoking"] = smoking_map[smoking]

    drinking_options = [
        "Tidak Pernah Minum",
        "Minum Sesekali",
        "Peminum Rutin",
    ]
    drinking_map = {
        "Tidak Pernah Minum": 3,
        "Minum Sesekali": 2,
        "Peminum Rutin": 1,
    }
    drinking_default_value_text = [
        k for k, v in drinking_map.items() if v == default_values["drinking"]
    ][0]
    drinking_default_index = drinking_options.index(drinking_default_value_text)
    drinking = st.selectbox(
        "Tingkat Konsumsi Alkohol",
        drinking_options,
        index=drinking_default_index,
        help="Pilih tingkat konsumsi alkohol pasien.",
    )
    input_data["drinking"] = drinking_map[drinking]


with col2:

    input_data["Chol"] = st.number_input(
        "Kolesterol Total (mmol/L)",
        min_value=1.0,
        max_value=15.0,
        value=default_values["Chol"],
        step=0.1,
        help="Jumlah total kolesterol dalam darah (mmol/L).",
    )

    input_data["HDL"] = st.number_input(
        "Kolesterol HDL (mmol/L)",
        min_value=0.1,
        max_value=4.0,
        value=default_values["HDL"],
        step=0.1,
        help="Kolesterol 'baik' (High-Density Lipoprotein) (mmol/L).",
    )

    input_data["LDL"] = st.number_input(
        "Kolesterol LDL (mmol/L)",
        min_value=0.1,
        max_value=10.0,
        value=default_values["LDL"],
        step=0.1,
        help="Kolesterol 'jahat' (Low-Density Lipoprotein) (mmol/L).",
    )

    input_data["FPG"] = st.number_input(
        "Gula Darah Puasa (FPG) (mmol/L)",
        min_value=1.0,
        max_value=25.0,
        value=default_values["FPG"],
        step=0.1,
        help="Tingkat glukosa darah setelah puasa minimal 8 jam (mmol/L).",
    )

    input_data["FFPG"] = st.number_input(
        "GDP Kunjungan Akhir (FFPG) (mmol/L)",
        min_value=1.0,
        max_value=30.0,
        value=default_values["FFPG"],
        step=0.1,
        help="Nilai GDP pada kunjungan klinis terakhir (mmol/L).",
    )

    input_data["BUN"] = st.number_input(
        "Blood Urea Nitrogen (BUN) (mmol/L)",
        min_value=1.0,
        max_value=50.0,
        value=default_values["BUN"],
        step=0.1,
        help="Blood Urea Nitrogen, indikator fungsi ginjal (mmol/L).",
    )


with col3:

    input_data["ALT"] = st.number_input(
        "Alanine Aminotransferase (ALT) (U/L)",
        min_value=0.0,
        max_value=500.0,
        value=default_values["ALT"],
        step=0.1,
        help="Alanine Aminotransferase, enzim hati (U/L).",
    )

    input_data["AST"] = st.number_input(
        "Aspartate Aminotransferase (AST) (U/L)",
        min_value=0.0,
        max_value=500.0,
        value=default_values["AST"],
        step=0.1,
        help="Aspartate Aminotransferase, enzim hati dan otot (U/L).",
    )

    input_data["SBP"] = st.number_input(
        "Tekanan Darah Sistolik (SBP) (mmHg)",
        min_value=50.0,
        max_value=250.0,
        value=default_values["SBP"],
        step=1.0,
        help="Tekanan Darah Sistolik, tekanan saat jantung berdetak (mmHg).",
    )

    input_data["DBP"] = st.number_input(
        "Tekanan Darah Diastolik (DBP) (mmHg)",
        min_value=30.0,
        max_value=150.0,
        value=default_values["DBP"],
        step=1.0,
        help="Tekanan Darah Diastolik, tekanan saat jantung istirahat (mmHg).",
    )

    input_data["Tri"] = st.number_input(
        "Trigliserida (mmol/L)",
        min_value=0.1,
        max_value=20.0,
        value=default_values["Tri"],
        step=0.1,
        help="Jenis lemak dalam darah (mmol/L).",
    )

    input_data["CCR"] = st.number_input(
        "Kreatinin Serum (µmol/L)",
        min_value=10.0,
        max_value=1000.0,
        value=default_values["CCR"],
        step=0.1,
        help="Tingkat kreatinin dalam darah, indikator fungsi ginjal (µmol/L). Jika fitur asli adalah CCR (mL/min), sesuaikan label dan rentang.",
    )
st.write("---")


if st.button("Prediksi Status Diabetes"):
    if not all_models_loaded:
        st.error("Beberapa model gagal dimuat. Tidak dapat melanjutkan.")
    elif scaler is None:
        st.error("Scaler gagal dimuat. Tidak dapat melanjutkan.")
    else:
        try:
            input_df = pd.DataFrame([input_data])[features]
            st.subheader("Data Yang Di Input")
            st.dataframe(input_df, hide_index=True)

            input_scaled = scaler.transform(input_df)
            st.subheader("Hasil Prediksi Model")
            results = {}

            for model_name, model in models.items():
                try:
                    pred = model.predict(input_scaled)[0]
                    if pred == 1:
                        prob = model.predict_proba(input_scaled)[0][1]
                    else:
                        prob = model.predict_proba(input_scaled)[0][0]
                    label = "Diabetes" if pred == 1 else "Tidak Diabetes"
                    results[model_name] = (label, prob)
                except Exception as e:
                    st.error(f"Kesalahan prediksi untuk {model_name}: {e}")
                    results[model_name] = ("Kesalahan", None)

            cols = st.columns(len(results))
            for i, (name, (label, prob)) in enumerate(results.items()):
                with cols[i]:
                    st.markdown(
                        """
                    <style>
                    [data-testid="stMetricValue"] {
                        font-size: 25px;
                        font-weight: bold;
                    }
                    </style>
                    """,
                        unsafe_allow_html=True,
                    )

                    st.metric(name, label)
                    if prob is not None:
                        st.write(f"Probabilitas: {prob * 100:.2f}%")
                    else:
                        st.write("Tidak tersedia")

        except Exception as e:
            st.error(f"Kesalahan prediksi: {e}")
