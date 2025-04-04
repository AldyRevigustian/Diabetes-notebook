import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

DATA_FILEPATH = "dataset_cleaned_oversampling.csv"
GENDER_MAP = {1: "Pria", 2: "Wanita"}
SMOKING_MAP = {3: "Tidak Pernah Merokok", 2: "Pernah Merokok", 1: "Perokok Aktif"}
DRINKING_MAP = {3: "Tidak Pernah Minum", 2: "Minum Sesekali", 1: "Peminum Rutin"}
FAMILY_HISTORY_MAP = {0: "Tidak", 1: "Ya"}
DIABETES_MAP = {0: "Non-Diabetes", 1: "Diabetes"}

NUMERIC_COLS_FOR_CONVERSION = [
    "Age",
    "BMI",
    "FPG",
    "SBP",
    "DBP",
    "Chol",
    "Tri",
    "HDL",
    "LDL",
    "ALT",
    "AST",
    "BUN",
    "CCR",
    "FFPG",
]

CORRELATION_COLS = [
    "Age",
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
    "Diabetes"
]

CATEGORICAL_COLS_MAPPED = [
    "Gender_label",
    "smoking_status",
    "drinking_status",
    "family_history_label",
    "Age_Group",
    "BMI_Category",
    "BP_Category",
    "FPG_Category",
]


COLOR_NON_DIABETIC = "#fcbba1"
COLOR_DIABETIC = "#cb181d"

DIABETES_COLOR_MAP = {"Non-Diabetes": COLOR_NON_DIABETIC, "Diabetes": COLOR_DIABETIC}

REDS_COLOR_SCALE = px.colors.sequential.Reds
REDS_DISCRETE_SEQUENCE = px.colors.sequential.Reds
SINGLE_LINE_COLOR = COLOR_DIABETIC


st.set_page_config(layout="wide", page_title="📊 Dashboard Diabetes")


def create_bmi_category(bmi):
    """Mengategorikan nilai BMI."""
    if pd.isna(bmi):
        return "Tidak Diketahui"
    if bmi < 18.5:
        return "Berat Badan Kurang"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Berat Badan Berlebih"
    return "Obesitas"


def create_fpg_category(fpg):
    """Mengategorikan nilai Fasting Plasma Glucose (FPG) / Gula Darah Puasa."""
    if pd.isna(fpg):
        return "Tidak Diketahui"
    if fpg < 5.6:
        return "Normal"
    if fpg < 7.0:
        return "Pra-diabetes (GDP Terganggu)"
    return "Ambang Batas Diabetes (>=7.0)"


def create_bp_category(sbp, dbp):
    """Mengategorikan Tekanan Darah (BP) berdasarkan SBP dan DBP."""
    if pd.isna(sbp) or pd.isna(dbp):
        return "Tidak Diketahui"
    if sbp < 120 and dbp < 80:
        return "Normal"
    if 120 <= sbp < 130 and dbp < 80:
        return "Meningkat"
    if 130 <= sbp < 140 or 80 <= dbp < 90:
        return "Hipertensi Tahap 1"
    if sbp >= 140 or dbp >= 90:
        return "Hipertensi Tahap 2"
    return "Tidak Terdefinisi"


def safe_calculate_prevalence(group):
    """Menghitung persentase prevalensi dengan aman dalam grup pandas."""
    total = group["Diabetes"].count()
    diabetic = group["Diabetes"].sum()
    prevalence = (diabetic / total * 100) if total > 0 else 0
    return pd.Series(
        {
            "Jumlah Diabetes": diabetic,
            "Jumlah Total": total,
            "Prevalensi (%)": prevalence,
        }
    )


@st.cache_data
def load_and_prepare_data(filepath):
    """Memuat, membersihkan, memetakan, dan merekayasa fitur untuk dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(
            f"Error: File tidak ditemukan di '{filepath}'. Pastikan file ada di direktori yang benar."
        )
        st.stop()

    df["Diabetes"] = pd.to_numeric(df["Diabetes"], errors="coerce")
    df.dropna(subset=["Diabetes"], inplace=True)
    df["Diabetes"] = df["Diabetes"].astype(int)

    for col in NUMERIC_COLS_FOR_CONVERSION:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:

            if col not in ["dataset"]:
                df[col] = np.nan

    df["Gender_label"] = df["Gender"].map(GENDER_MAP).fillna("Tidak Diketahui")
    df["smoking_status"] = df["smoking"].map(SMOKING_MAP).fillna("Tidak Diketahui")
    df["drinking_status"] = df["drinking"].map(DRINKING_MAP).fillna("Tidak Diketahui")
    df["family_history_label"] = (
        df["family_history"].map(FAMILY_HISTORY_MAP).fillna("Tidak Diketahui")
    )
    df["Diabetes_label"] = df["Diabetes"].map(DIABETES_MAP)

    max_age = df["Age"].max()
    if pd.notna(max_age):
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 30, 40, 50, 60, 70, int(max_age) + 1],
            labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"],
            right=False,
        ).astype(str)
    else:
        df["Age_Group"] = "Tidak Diketahui"

    df["BMI_Category"] = df["BMI"].apply(create_bmi_category)
    df["FPG_Category"] = df["FPG"].apply(create_fpg_category)
    df["BP_Category"] = df.apply(
        lambda row: create_bp_category(row.get("SBP"), row.get("DBP")), axis=1
    )

    return df


df_full = load_and_prepare_data(DATA_FILEPATH)
df = df_full.copy()

st.sidebar.header("Kontrol Dashboard")


dataset_options = ["Semua", "train", "validation", "test"]
selected_dataset = st.sidebar.selectbox("Pilih Set Dataset", dataset_options)
if selected_dataset != "Semua" and "dataset" in df.columns:
    df = df[df["dataset"] == selected_dataset]


gender_options = ["Semua"] + sorted(
    [g for g in df_full["Gender_label"].unique() if g != "Tidak Diketahui"]
)
gender_multiselect_options = sorted([g for g in gender_options if g not in ["Semua"]])
default_genders = gender_multiselect_options if gender_multiselect_options else []
selected_genders = st.sidebar.multiselect(
    "Pilih Jenis Kelamin",
    options=gender_multiselect_options,
    default=default_genders,
)
if selected_genders:
    df = df[df["Gender_label"].isin(selected_genders)]
elif len(gender_multiselect_options) > 0:
    df = df.iloc[0:0]


min_age_full = df_full["Age"].min()
max_age_full = df_full["Age"].max()
if pd.notna(min_age_full) and pd.notna(max_age_full):
    selected_age = st.sidebar.slider(
        "Pilih Rentang Usia",
        int(min_age_full),
        int(max_age_full),
        (int(min_age_full), int(max_age_full)),
    )
    df = df[df["Age"].between(selected_age[0], selected_age[1])]
else:
    st.sidebar.warning(
        "Data Usia hilang atau tidak valid, tidak dapat memfilter berdasarkan usia."
    )


min_bmi_full = df_full["BMI"].min()
max_bmi_full = df_full["BMI"].max()
if pd.notna(min_bmi_full) and pd.notna(max_bmi_full):
    selected_bmi = st.sidebar.slider(
        "Pilih Rentang BMI",
        float(min_bmi_full),
        float(max_bmi_full),
        (float(min_bmi_full), float(max_bmi_full)),
        step=0.1,
    )
    df = df[df["BMI"].between(selected_bmi[0], selected_bmi[1])]
else:
    st.sidebar.warning(
        "Data BMI hilang atau tidak valid, tidak dapat memfilter berdasarkan BMI."
    )


if df.empty:
    st.error(
        "Tidak ada data yang cocok dengan filter yang dipilih. Silakan sesuaikan kontrol di sidebar."
    )
    st.stop()


st.title("📊 Dashboard Diabetes")

total_patients = len(df)
diabetic_patients = int(df["Diabetes"].sum())
non_diabetic_patients = total_patients - diabetic_patients
prevalence = (diabetic_patients / total_patients * 100) if total_patients > 0 else 0

st.markdown(
    f"Menganalisis **{total_patients:,}** catatan pasien berdasarkan filter saat ini."
)
if "dataset" in df.columns:
    dataset_label = selected_dataset if selected_dataset != "Semua" else "Semua Data"
    st.markdown(f"Set Dataset: **{dataset_label}**")
st.markdown("---")

st.header("Overview Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Pasien", f"{total_patients:,}")
col2.metric("Pasien Diabetes", f"{diabetic_patients:,}")
col3.metric("Pasien Non-Diabetes", f"{non_diabetic_patients:,}")
col4.metric("Prevalensi Diabetes", f"{prevalence:.1f}%")


if (
    total_patients > 0
    and "Diabetes_label" in df.columns
    and df["Diabetes_label"].nunique() > 0
):
    if df["Diabetes_label"].nunique() > 1:
        fig_dist = px.pie(
            df,
            names="Diabetes_label",
            title="Distribusi Data Diabetes",
            color="Diabetes_label",
            color_discrete_map=DIABETES_COLOR_MAP,
            hole=0.3,
        )

        pull_values = [
            0.05 if label == "Diabetes" else 0 for label in fig_dist.data[0].labels
        ]
        fig_dist.update_traces(textinfo="percent+label", pull=pull_values)
        fig_dist.update_layout(legend_title_text="Status Diabetes")
        st.plotly_chart(fig_dist, use_container_width=True)
    elif df["Diabetes_label"].nunique() == 1:
        unique_label = df["Diabetes_label"].dropna().iloc[0]
        st.info(f"Data terfilter hanya berisi pasien **{unique_label}**.")
else:
    st.warning(
        "Tidak ada data yang tersedia untuk distribusi Status Diabetes dengan filter saat ini."
    )
st.markdown("---")

st.header("Categorical Features vs Diabetes")
categorical_cols_for_dist = [
    "Gender_label",
    "smoking_status",
    "drinking_status",
    "family_history_label",
    "Age_Group",
    "BMI_Category",
    "BP_Category",
    "FPG_Category",
]
available_cats_dist = [
    col
    for col in categorical_cols_for_dist
    if col in df.columns and df[col].nunique() > 0
]

if available_cats_dist:
    cat_feature_to_plot = st.selectbox(
        "Pilih Fitur Kategorikal",
        available_cats_dist,
        index=(
            available_cats_dist.index("Age_Group")
            if "Age_Group" in available_cats_dist
            else (
                available_cats_dist.index("BMI_Category")
                if "BMI_Category" in available_cats_dist
                else 0
            )
        ),
    )
    if cat_feature_to_plot and total_patients > 0:

        category_orders = {}
        if cat_feature_to_plot == "Age_Group":
            category_orders = {
                "<30": 0,
                "30-39": 1,
                "40-49": 2,
                "50-59": 3,
                "60-69": 4,
                "70+": 5,
                "Tidak Diketahui": 6,
            }
        elif cat_feature_to_plot == "BMI_Category":
            category_orders = {
                "Berat Badan Kurang": 0,
                "Normal": 1,
                "Berat Badan Berlebih": 2,
                "Obesitas": 3,
                "Tidak Diketahui": 4,
            }
        elif cat_feature_to_plot == "BP_Category":
            category_orders = {
                "Normal": 0,
                "Meningkat": 1,
                "Hipertensi Tahap 1": 2,
                "Hipertensi Tahap 2": 3,
                "Tidak Diketahui": 4,
                "Tidak Terdefinisi": 5,
            }
        elif cat_feature_to_plot == "FPG_Category":
            category_orders = {
                "Normal": 0,
                "Pra-diabetes (GDP Terganggu)": 1,
                "Ambang Batas Diabetes (>=7.0)": 2,
                "Tidak Diketahui": 3,
            }

        present_cats = df[cat_feature_to_plot].unique()
        category_orders = {
            k: v for k, v in category_orders.items() if k in present_cats
        }

        fig_cat = px.histogram(
            df,
            x=cat_feature_to_plot,
            color="Diabetes_label",
            title=f"Distribusi {cat_feature_to_plot.replace('_label','').replace('_',' ').title()} berdasarkan Status Diabetes",
            barmode="group",
            color_discrete_map=DIABETES_COLOR_MAP,
            category_orders=(
                {cat_feature_to_plot: sorted(category_orders, key=category_orders.get)}
                if category_orders
                else None
            ),
            labels={"Diabetes_label": "Status Diabetes"},
        )
        fig_cat.update_layout(
            xaxis_title=cat_feature_to_plot.replace("_label", "")
            .replace("_", " ")
            .title(),
            yaxis_title="Jumlah",
        )
        st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.warning(
        "Tidak ada fitur kategorikal yang tersedia untuk analisis distribusi dengan filter saat ini."
    )
st.markdown("---")


st.header("Prevalensi Diabetes dalam Kategori")
available_cats_prev = [
    col
    for col in CATEGORICAL_COLS_MAPPED
    if col in df.columns and df[col].nunique() > 1
]

if available_cats_prev:
    cat_feature_for_prevalence = st.selectbox(
        "Pilih Kategori untuk Analisis Prevalensi",
        available_cats_prev,
        key="prevalence_cat",
        index=(
            available_cats_prev.index("Age_Group")
            if "Age_Group" in available_cats_prev
            else (
                available_cats_prev.index("BMI_Category")
                if "BMI_Category" in available_cats_prev
                else 0
            )
        ),
    )
    if cat_feature_for_prevalence and total_patients > 0:

        prevalence_df = (
            df.groupby(cat_feature_for_prevalence)
            .apply(safe_calculate_prevalence)
            .reset_index()
        )

        ordered_categories_prev = {}
        if cat_feature_for_prevalence == "Age_Group":
            ordered_categories_prev = {
                "<30": 0,
                "30-39": 1,
                "40-49": 2,
                "50-59": 3,
                "60-69": 4,
                "70+": 5,
                "Tidak Diketahui": 6,
            }
        elif cat_feature_for_prevalence == "BMI_Category":
            ordered_categories_prev = {
                "Berat Badan Kurang": 0,
                "Normal": 1,
                "Berat Badan Berlebih": 2,
                "Obesitas": 3,
                "Tidak Diketahui": 4,
            }
        elif cat_feature_for_prevalence == "BP_Category":
            ordered_categories_prev = {
                "Normal": 0,
                "Meningkat": 1,
                "Hipertensi Tahap 1": 2,
                "Hipertensi Tahap 2": 3,
                "Tidak Diketahui": 4,
                "Tidak Terdefinisi": 5,
            }
        elif cat_feature_for_prevalence == "FPG_Category":
            ordered_categories_prev = {
                "Normal": 0,
                "Pra-diabetes (GDP Terganggu)": 1,
                "Ambang Batas Diabetes (>=7.0)": 2,
                "Tidak Diketahui": 3,
            }

        if ordered_categories_prev:
            present_cats_prev = prevalence_df[cat_feature_for_prevalence].unique()
            ordered_categories_prev = {
                k: v
                for k, v in ordered_categories_prev.items()
                if k in present_cats_prev
            }
            prevalence_df["sort_order"] = prevalence_df[cat_feature_for_prevalence].map(
                ordered_categories_prev
            )
            prevalence_df = prevalence_df.sort_values("sort_order").drop(
                "sort_order", axis=1
            )

        fig_prev = px.bar(
            prevalence_df,
            x=cat_feature_for_prevalence,
            y="Prevalensi (%)",
            title=f"Prevalensi Diabetes dalam setiap Kategori '{cat_feature_for_prevalence.replace('_label','').replace('_',' ').title()}'",
            color="Prevalensi (%)",
            color_continuous_scale=REDS_COLOR_SCALE,
            text_auto=".1f",
            labels={"Prevalensi (%)": "Prevalensi Diabetes (%)"},
        )

        fig_prev.update_traces(marker_coloraxis="coloraxis")
        fig_prev.update_traces(textangle=0, textposition="outside")
        fig_prev.update_layout(
            xaxis_title=cat_feature_for_prevalence.replace("_label", "")
            .replace("_", " ")
            .title(),
            yaxis_title="Prevalensi Diabetes (%)",
            yaxis_ticksuffix="%",
        )
        st.plotly_chart(fig_prev, use_container_width=True)

        with st.expander("Tampilkan Tabel Data Prevalensi"):
            st.dataframe(
                prevalence_df[
                    [
                        cat_feature_for_prevalence,
                        "Jumlah Diabetes",
                        "Jumlah Total",
                        "Prevalensi (%)",
                    ]
                ].round(1)
            )
else:
    st.warning(
        "Tidak ada fitur kategorikal yang cocok tersedia untuk analisis prevalensi dengan filter saat ini (membutuhkan >1 kategori)."
    )
st.markdown("---")


st.header("Distribusi Fitur Numerik & Ambang Batas Klinis")

numerical_cols = sorted(
    [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col
        not in [
            "Diabetes",
            "Gender",
            "smoking",
            "drinking",
            "family_history",
        ]
        and df[col].nunique() > 1
    ]
)

if not numerical_cols:
    st.warning(
        "Tidak ada fitur numerik dengan varians yang tersedia untuk analisis distribusi dengan filter saat ini."
    )
else:

    default_feature_idx = 0
    if "FPG" in numerical_cols:
        default_feature_idx = numerical_cols.index("FPG")
    elif "BMI" in numerical_cols:
        default_feature_idx = numerical_cols.index("BMI")
    feature_to_plot = st.selectbox(
        "Pilih Fitur Numerik untuk Divisualisasikan",
        numerical_cols,
        index=default_feature_idx,
        key="dist_feature_select",
    )

    if feature_to_plot and total_patients > 0:

        hist_args = dict(
            x=feature_to_plot,
            color="Diabetes_label",
            marginal="box",
            color_discrete_map=DIABETES_COLOR_MAP,
            barmode="overlay",
            opacity=0.7,
            labels={"Diabetes_label": "Status Diabetes"},
        )

        fig_dist_num = px.histogram(
            df,
            **hist_args,
            title=f"Histogram {feature_to_plot} berdasarkan Status Diabetes",
        )
        fig_dist_num.update_layout(xaxis_title=feature_to_plot)
        st.plotly_chart(fig_dist_num, use_container_width=True)

        fig_box = px.box(
            df,
            x="Diabetes_label",
            y=feature_to_plot,
            color="Diabetes_label",
            title=f"Box Plot {feature_to_plot} berdasarkan Status Diabetes",
            color_discrete_map=DIABETES_COLOR_MAP,
            labels={
                "Diabetes_label": "Status Diabetes",
                feature_to_plot: feature_to_plot,
            },
            points="outliers",
            notched=True,
        )
        st.plotly_chart(fig_box, use_container_width=True)
st.markdown("---")


st.header("Analisis Bivariat (Scatter Plot)")
if len(numerical_cols) < 2:
    st.warning(
        "Dibutuhkan setidaknya dua fitur numerik dengan varians untuk analisis scatter plot dengan filter saat ini."
    )
else:
    col_scatter1, col_scatter2 = st.columns(2)

    default_x_idx = numerical_cols.index("BMI") if "BMI" in numerical_cols else 0
    default_y_idx = (
        numerical_cols.index("FPG")
        if "FPG" in numerical_cols
        else (1 if len(numerical_cols) > 1 else 0)
    )

    if default_x_idx == default_y_idx and len(numerical_cols) > 1:
        default_y_idx = 1

    x_axis = col_scatter1.selectbox(
        "Pilih Fitur Sumbu X", numerical_cols, index=default_x_idx, key="scatter_x"
    )
    y_axis = col_scatter2.selectbox(
        "Pilih Fitur Sumbu Y", numerical_cols, index=default_y_idx, key="scatter_y"
    )

    if x_axis and y_axis and x_axis != y_axis and total_patients > 0:

        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="Diabetes_label",
            title=f"{y_axis} vs. {x_axis} berdasarkan Status Diabetes",
            color_discrete_map=DIABETES_COLOR_MAP,
            hover_data=[
                col
                for col in ["Age", "Gender_label", "BMI", "FPG", "SBP", "DBP"]
                if col in df.columns
            ],
            labels={"Diabetes_label": "Status Diabetes"},
            opacity=0.7,
        )
        fig_scatter.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
        st.plotly_chart(fig_scatter, use_container_width=True)
    elif x_axis == y_axis:
        st.warning("Silakan pilih fitur yang berbeda untuk sumbu X dan Y.")
st.markdown("---")


st.header("Matriks Korelasi Fitur Numerik")

corr_cols_available = [
    col for col in CORRELATION_COLS if col in df.columns and df[col].nunique() > 1
]

if len(corr_cols_available) > 1 and total_patients > 1:

    corr_subset = st.radio(
        "Hitung Korelasi Untuk:",
        ("Semua Data Terfilter", "Hanya Diabetes", "Hanya Non-Diabetes"),
        horizontal=True,
        key="corr_radio",
    )
    df_corr_subset = df.copy()
    if corr_subset == "Hanya Diabetes":
        df_corr_subset = df[df["Diabetes"] == 1].copy()
    elif corr_subset == "Hanya Non-Diabetes":
        df_corr_subset = df[df["Diabetes"] == 0].copy()

    corr_cols_final = [
        col
        for col in corr_cols_available
        if col in df_corr_subset.columns and df_corr_subset[col].nunique() > 1
    ]

    if len(df_corr_subset) > 1 and len(corr_cols_final) > 1:
        corr_method = "spearman"

        corr_matrix = df_corr_subset[corr_cols_final].corr(method=corr_method)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title=f"Matriks Korelasi",
            color_continuous_scale=REDS_COLOR_SCALE,
        )
        fig_corr.update_xaxes(side="bottom")
        fig_corr.update_layout(coloraxis_colorbar=dict(title="Korelasi"))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning(
            f"Data tidak cukup (perlu > 1 baris dan > 1 kolom dengan varians) dalam grup '{corr_subset}' untuk perhitungan korelasi dengan filter saat ini."
        )
else:
    st.warning(
        "Kolom numerik dengan varians (perlu > 1) atau titik data (perlu > 1) tidak cukup untuk matriks korelasi berdasarkan filter saat ini."
    )
st.markdown("---")


st.header("Trend Prevalensi Diabetes")

grouping_options_ordered = ["Age_Group", "BMI_Category", "BP_Category", "FPG_Category"]

available_groupings = [
    opt
    for opt in grouping_options_ordered
    if opt in df.columns and df[opt].nunique() > 1
]

if available_groupings:

    x_grouping = st.selectbox(
        "Kelompokkan Prevalensi Berdasarkan (Sumbu X)",
        available_groupings,
        index=(
            available_groupings.index("Age_Group")
            if "Age_Group" in available_groupings
            else 0
        ),
        key="line_grouping",
    )

    segment_options_all = [
        "Gender_label",
        "smoking_status",
        "drinking_status",
        "family_history_label",
    ]

    available_segments = ["Tidak Ada"] + [
        opt
        for opt in segment_options_all
        if opt in df.columns and opt != x_grouping and df[opt].nunique() > 1
    ]
    segment_by = st.selectbox(
        "Segmentasi Garis Berdasarkan (Opsional)",
        available_segments,
        key="line_segment",
    )

    if x_grouping and total_patients > 0:

        ordered_categories = {}
        if x_grouping == "Age_Group":
            ordered_categories = {
                "<30": 0,
                "30-39": 1,
                "40-49": 2,
                "50-59": 3,
                "60-69": 4,
                "70+": 5,
                "Tidak Diketahui": 6,
            }
        elif x_grouping == "BMI_Category":
            ordered_categories = {
                "Berat Badan Kurang": 0,
                "Normal": 1,
                "Berat Badan Berlebih": 2,
                "Obesitas": 3,
                "Tidak Diketahui": 4,
            }
        elif x_grouping == "BP_Category":
            ordered_categories = {
                "Normal": 0,
                "Meningkat": 1,
                "Hipertensi Tahap 1": 2,
                "Hipertensi Tahap 2": 3,
                "Tidak Diketahui": 4,
                "Tidak Terdefinisi": 5,
            }
        elif x_grouping == "FPG_Category":
            ordered_categories = {
                "Normal": 0,
                "Pra-diabetes (GDP Terganggu)": 1,
                "Ambang Batas Diabetes (>=7.0)": 2,
                "Tidak Diketahui": 3,
            }

        grouping_cols = [x_grouping]
        if segment_by != "Tidak Ada":
            grouping_cols.append(segment_by)

        grouped_data = (
            df.groupby(grouping_cols).apply(safe_calculate_prevalence).reset_index()
        )

        grouped_data = grouped_data[grouped_data["Jumlah Total"] > 0]

        if ordered_categories:
            present_cats_line = grouped_data[x_grouping].unique()
            current_order = {
                k: v for k, v in ordered_categories.items() if k in present_cats_line
            }
            grouped_data["sort_order"] = grouped_data[x_grouping].map(current_order)
            sort_cols = (
                [segment_by, "sort_order"]
                if segment_by != "Tidak Ada"
                else ["sort_order"]
            )
            grouped_data = grouped_data.sort_values(sort_cols).drop(
                "sort_order", axis=1
            )
        elif segment_by != "Tidak Ada":
            grouped_data = grouped_data.sort_values([segment_by, x_grouping])
        else:
            grouped_data = grouped_data.sort_values(x_grouping)

        plot_title = f"Prevalensi Diabetes berdasarkan {x_grouping.replace('_label','').replace('_', ' ').title()}"
        color_col = segment_by if segment_by != "Tidak Ada" else None

        color_args = {}
        if segment_by == "Tidak Ada":

            color_args["color_discrete_sequence"] = [SINGLE_LINE_COLOR]
        else:

            plot_title += (
                f" dan {segment_by.replace('_label','').replace('_', ' ').title()}"
            )
            color_args["color"] = segment_by

            color_args["color_discrete_sequence"] = REDS_DISCRETE_SEQUENCE[3:]

        fig_line = px.line(
            grouped_data,
            x=x_grouping,
            y="Prevalensi (%)",
            title=plot_title,
            markers=True,
            line_shape="linear",
            labels={"Prevalensi (%)": "Prevalensi Diabetes (%)"},
            hover_data=["Jumlah Diabetes", "Jumlah Total"],
            **color_args,
        )

        fig_line.update_layout(
            xaxis_title=x_grouping.replace("_label", "").replace("_", " ").title(),
            yaxis_title="Prevalensi Diabetes (%)",
            yaxis_ticksuffix="%",
            legend_title=(
                segment_by.replace("_label", "").replace("_", " ").title()
                if segment_by != "Tidak Ada"
                else None
            ),
        )

        if segment_by == "Tidak Ada":

            fig_line.update_traces(
                text=grouped_data["Prevalensi (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="top center",
                mode="lines+markers+text",
                line=dict(width=2.5),
            )
        else:

            fig_line.update_traces(mode="lines+markers", line=dict(width=3))

        st.plotly_chart(fig_line, use_container_width=True)

        with st.expander("Tampilkan Tabel Data Tren"):
            display_cols = grouping_cols + [
                "Jumlah Diabetes",
                "Jumlah Total",
                "Prevalensi (%)",
            ]
            st.dataframe(grouped_data[display_cols].round(1))
else:
    st.warning(
        "Tidak ada variabel kategorikal terurut yang cocok tersedia untuk analisis grafik garis dengan filter saat ini."
    )
st.markdown("---")


st.header("Data Explorer")
st.dataframe(df)


csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Unduh Data sebagai CSV",
    data=csv,
    file_name="diabetes_data.csv",
    mime="text/csv",
)
