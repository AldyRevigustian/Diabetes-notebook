import streamlit as st

st.set_page_config(page_title="🧾 Glosarium Dataset", layout="wide")

st.title("🧾 Glosarium Dataset")
st.markdown(
    "Halaman ini menyediakan definisi dan satuan untuk semua fitur yang digunakan dalam aplikasi prediksi dan di dalam dataset diabetes."
)
st.markdown("---")


st.markdown("### Usia (Age)")
st.write("Usia pasien dalam tahun pada saat pengumpulan data (diukur dalam **tahun**).")
st.markdown("---")

st.markdown("### Jenis Kelamin (Gender)")
st.write("Jenis kelamin biologis pasien:")
st.markdown("- **1:** Pria")
st.markdown("- **2:** Wanita")
st.markdown("---")

st.markdown("### IMT (Indeks Massa Tubuh / BMI)")
st.write(
    "Indeks Massa Tubuh dihitung sebagai berat badan dibagi kuadrat tinggi badan (diukur dalam **kg/m²**). Digunakan untuk mengklasifikasikan berat badan kurang, normal, berlebih, atau obesitas."
)
# Penjelasan BMI_Category
st.markdown("##### Kategori IMT (`BMI_Category`)")
st.write("Berdasarkan nilai IMT, pasien dikategorikan lebih lanjut menjadi:")
st.markdown("- **Berat Badan Kurang:** IMT < 18.5")
st.markdown("- **Normal:** 18.5 ≤ IMT < 25")
st.markdown("- **Berat Badan Berlebih:** 25 ≤ IMT < 30")
st.markdown("- **Obesitas:** IMT ≥ 30")
st.markdown("---")

st.markdown("### TDS (Tekanan Darah Sistolik / SBP)")
st.write("Tekanan di arteri saat jantung berdetak (diukur dalam **mmHg**).")
st.markdown("---")

st.markdown("### TDD (Tekanan Darah Diastolik / DBP)")
st.write(
    "Tekanan di arteri saat jantung beristirahat di antara detak (diukur dalam **mmHg**)."
)
# Penjelasan BP_Category (setelah SBP dan DBP)
st.markdown("##### Kategori Tekanan Darah (`BP_Category`)")
st.write("Berdasarkan nilai TDS dan TDD, tekanan darah pasien dikategorikan menjadi:")
st.markdown("- **Normal:** TDS < 120 mmHg DAN TDD < 80 mmHg")
st.markdown("- **Meningkat:** 120 ≤ TDS < 130 mmHg DAN TDD < 80 mmHg")
st.markdown("- **Hipertensi Tahap 1:** 130 ≤ TDS < 140 mmHg ATAU 80 ≤ TDD < 90 mmHg")
st.markdown("- **Hipertensi Tahap 2:** TDS ≥ 140 mmHg ATAU TDD ≥ 90 mmHg")
st.markdown("---")

st.markdown("### GDP (Gula Darah Puasa / FPG)")
st.write(
    "Tingkat glukosa darah diukur setelah puasa minimal 8 jam (diukur dalam **mmol/L**)."
)
# Penjelasan FPG_Category
st.markdown("##### Kategori GDP (`FPG_Category`)")
st.write("Berdasarkan nilai GDP, status glikemik pasien dikategorikan menjadi:")
st.markdown("- **Normal:** GDP < 5.6 mmol/L")
st.markdown("- **Pra-diabetes (GDP Terganggu):** 5.6 ≤ GDP < 7.0 mmol/L")
st.markdown("- **Ambang Batas Diabetes (≥7.0):** GDP ≥ 7.0 mmol/L")
st.markdown("---")

st.markdown("### GDP Kunjungan Akhir (FFPG)")
st.write(
    "Gula Darah Puasa terakhir yang diukur pada kunjungan klinis terakhir (diukur dalam **mmol/L**)."
)
st.markdown("---")

st.markdown("### Kolesterol Total (Chol)")
st.write("Jumlah total kolesterol dalam darah (diukur dalam **mmol/L**).")
st.markdown("---")

st.markdown("### Trigliserida (Tri)")
st.write(
    "Jenis lemak (lipid) dalam darah. Kadar tinggi dapat meningkatkan risiko penyakit jantung (diukur dalam **mmol/L**)."
)
st.markdown("---")

st.markdown("### HDL (High-Density Lipoprotein)")
st.write(
    "Kolesterol baik yang membantu menghilangkan bentuk kolesterol lain dari aliran darah (diukur dalam **mmol/L**)."
)
st.markdown("---")

st.markdown("### LDL (Low-Density Lipoprotein)")
st.write(
    "Kolesterol jahat yang dapat menumpuk dan menyumbat pembuluh darah (diukur dalam **mmol/L**)."
)
st.markdown("---")

st.markdown("### ALT (Alanine Aminotransferase)")
st.write(
    "Enzim hati. Kadar tinggi dapat mengindikasikan kerusakan atau penyakit hati (diukur dalam **U/L**)."
)
st.markdown("---")

st.markdown("### AST (Aspartate Aminotransferase)")
st.write(
    "Enzim yang ditemukan di hati dan jaringan lain. Kadar tinggi juga dapat mengindikasikan kerusakan hati atau otot (diukur dalam **U/L**)."
)
st.markdown("---")

st.markdown("### BUN (Blood Urea Nitrogen)")
st.write(
    "Jumlah nitrogen dalam darah yang berasal dari urea. Digunakan untuk mengevaluasi fungsi ginjal (diukur dalam **mmol/L**)."
)
st.markdown("---")

st.markdown("### Kreatinin Serum (CCR)")
st.write(
    "Kadar kreatinin dalam serum darah. Nilai ini digunakan untuk memperkirakan Laju Filtrasi Glomerulus (GFR) atau Klirens Kreatinin (CCR), yang merupakan penanda penting fungsi ginjal (diukur dalam **µmol/L**)."
)
st.markdown("---")

st.markdown("### Merokok (Smoking)")
st.write("Status merokok pasien:")
st.markdown("- **1:** Perokok Aktif")
st.markdown("- **2:** Pernah Merokok")
st.markdown("- **3:** Tidak Pernah Merokok")
st.markdown("---")

st.markdown("### Minum Alkohol (Drinking)")
st.write("Tingkat konsumsi alkohol pasien:")
st.markdown("- **1:** Peminum Rutin")
st.markdown("- **2:** Minum Sesekali")
st.markdown("- **3:** Tidak Pernah Minum")
st.markdown("---")

st.markdown("### Riwayat Keluarga (Family History)")
st.write("Menunjukkan apakah pasien memiliki riwayat diabetes dalam keluarga:")
st.markdown("- **0:** Tidak")
st.markdown("- **1:** Ya")
st.markdown("---")

st.markdown("### Diabetes (Variabel Target)")
st.write("Menunjukkan apakah pasien menderita diabetes:")
st.markdown("- **0:** Non-Diabetes")
st.markdown("- **1:** Diabetes")
st.markdown("---")
