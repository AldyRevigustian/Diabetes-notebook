import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

DATA_FILEPATH = "dataset_cleaned_oversampling.csv"
GENDER_MAP = {1: "Male", 2: "Female"}
SMOKING_MAP = {1: "Never Smoker", 2: "Former Smoker", 3: "Current Smoker"}
DRINKING_MAP = {1: "Never Drinker", 2: "Occasional Drinker", 3: "Regular Drinker"}
FAMILY_HISTORY_MAP = {0: "No", 1: "Yes"}
DIABETES_MAP = {0: "Non-Diabetic", 1: "Diabetic"}

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

# --- Color Theme ---

# px.colors.sequential.Reds for options:
# ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
COLOR_NON_DIABETIC = "#fcbba1"  
COLOR_DIABETIC = "#cb181d"  

DIABETES_COLOR_MAP = {"Non-Diabetic": COLOR_NON_DIABETIC, "Diabetic": COLOR_DIABETIC}

REDS_COLOR_SCALE = px.colors.sequential.Reds
REDS_DISCRETE_SEQUENCE = px.colors.sequential.Reds
SINGLE_LINE_COLOR = COLOR_DIABETIC


st.set_page_config(layout="wide", page_title="📊 Diabetes Dashboard")

def create_bmi_category(bmi):
    """Categorizes BMI values."""
    if pd.isna(bmi):
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def create_fpg_category(fpg):
    """Categorizes Fasting Plasma Glucose (FPG) values."""
    if pd.isna(fpg):
        return "Unknown"
    if fpg < 5.6:
        return "Normal"
    if fpg < 7.0:
        return "Pre-diabetes (IFG)"
    return "Diabetes Threshold (>=7.0)"


def create_bp_category(sbp, dbp):
    """Categorizes Blood Pressure (BP) based on SBP and DBP."""
    if pd.isna(sbp) or pd.isna(dbp):
        return "Unknown"
    if sbp < 120 and dbp < 80:
        return "Normal"
    if 120 <= sbp < 130 and dbp < 80:
        return "Elevated"
    if 130 <= sbp < 140 or 80 <= dbp < 90:
        return "Hypertension Stage 1"
    if sbp >= 140 or dbp >= 90:
        return "Hypertension Stage 2"
    return "Undefined"


def safe_calculate_prevalence(group):
    """Safely calculates prevalence percentage within a pandas group."""
    total = group["Diabetes"].count()
    diabetic = group["Diabetes"].sum()
    prevalence = (diabetic / total * 100) if total > 0 else 0
    return pd.Series(
        {"Diabetic Count": diabetic, "Total Count": total, "Prevalence (%)": prevalence}
    )

@st.cache_data
def load_and_prepare_data(filepath):
    """Loads, cleans, maps, and engineers features for the dataset."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(
            f"Error: File not found at '{filepath}'. Please ensure the file exists in the correct directory."
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

    df["Gender_label"] = df["Gender"].map(GENDER_MAP).fillna("Unknown")
    df["smoking_status"] = df["smoking"].map(SMOKING_MAP).fillna("Unknown")
    df["drinking_status"] = df["drinking"].map(DRINKING_MAP).fillna("Unknown")
    df["family_history_label"] = (
        df["family_history"].map(FAMILY_HISTORY_MAP).fillna("Unknown")
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
        df["Age_Group"] = "Unknown"

    df["BMI_Category"] = df["BMI"].apply(create_bmi_category)
    df["FPG_Category"] = df["FPG"].apply(create_fpg_category)
    df["BP_Category"] = df.apply(
        lambda row: create_bp_category(row.get("SBP"), row.get("DBP")), axis=1
    )

    return df

df_full = load_and_prepare_data(DATA_FILEPATH)
df = df_full.copy()

st.sidebar.header("Dashboard Controls")

dataset_options = ["All", "train", "validation", "test"]

selected_dataset = st.sidebar.selectbox("Select Dataset Split", dataset_options)
if selected_dataset != "All" and "dataset" in df.columns:
    df = df[df["dataset"] == selected_dataset]

gender_options = ["All"] + sorted(list(df_full["Gender_label"].unique()))
gender_multiselect_options = sorted(
    [g for g in gender_options if g not in ["All", "Unknown"]]
)
default_genders = gender_multiselect_options if gender_multiselect_options else []
selected_genders = st.sidebar.multiselect(
    "Select Gender(s)",
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
        "Select Age Range",
        int(min_age_full),
        int(max_age_full),
        (int(min_age_full), int(max_age_full)),
    )
    df = df[df["Age"].between(selected_age[0], selected_age[1])]
else:
    st.sidebar.warning("Age data missing or invalid, cannot filter by age.")

min_bmi_full = df_full["BMI"].min()
max_bmi_full = df_full["BMI"].max()
if pd.notna(min_bmi_full) and pd.notna(max_bmi_full):
    selected_bmi = st.sidebar.slider(
        "Select BMI Range",
        float(min_bmi_full),
        float(max_bmi_full),
        (float(min_bmi_full), float(max_bmi_full)),
        step=0.1,
    )
    df = df[df["BMI"].between(selected_bmi[0], selected_bmi[1])]
else:
    st.sidebar.warning("BMI data missing or invalid, cannot filter by BMI.")

# --- Filter Application Check ---
if df.empty:
    st.error(
        "No data matches the selected filters. Please adjust the controls in the sidebar."
    )
    st.stop()

st.title("📊 Diabetes Dashboard")

total_patients = len(df)
diabetic_patients = int(df["Diabetes"].sum())
non_diabetic_patients = total_patients - diabetic_patients
prevalence = (diabetic_patients / total_patients * 100) if total_patients > 0 else 0

# --- Header Information ---
st.markdown(
    f"Analyzing **{total_patients:,}** patient records based on current filters."
)
if "dataset" in df.columns:
    dataset_label = selected_dataset if selected_dataset != "All" else "All Data"
    st.markdown(f"Dataset Split: **{dataset_label}**")
st.markdown("---")

# --- Overview Section ---
st.header("Snapshot Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", f"{total_patients:,}")
col2.metric("Diabetic Patients", f"{diabetic_patients:,}")
col3.metric("Non-Diabetic Patients", f"{non_diabetic_patients:,}")
col4.metric("Diabetes Prevalence", f"{prevalence:.1f}%")

original_contains_train = (
    "train" in df_full["dataset"].unique() if "dataset" in df_full else False
)
current_df_contains_train = False
if "dataset" in df.columns:
    current_df_contains_train = "train" in df["dataset"].unique()
if original_contains_train and (
    selected_dataset == "train"
    or (selected_dataset == "All" and current_df_contains_train)
):
    ()

if (
    total_patients > 0
    and "Diabetes_label" in df.columns
    and df["Diabetes_label"].nunique() > 0
):
    if df["Diabetes_label"].nunique() > 1:
        fig_dist = px.pie(
            df,
            names="Diabetes_label",
            title="Diabetes Status Distribution (Filtered Data)",
            color="Diabetes_label",
            color_discrete_map=DIABETES_COLOR_MAP,  # Apply the defined Reds map
            hole=0.3,
        )
        pull_values = [
            0.05 if label == "Diabetic" else 0 for label in fig_dist.data[0].labels
        ]
        fig_dist.update_traces(textinfo="percent+label", pull=pull_values)
        fig_dist.update_layout(legend_title_text="Diabetes Status")
        st.plotly_chart(fig_dist, use_container_width=True)
    elif df["Diabetes_label"].nunique() == 1:
        unique_label = df["Diabetes_label"].dropna().iloc[0]
        st.info(f"Filtered data contains only **{unique_label}** patients.")
else:
    st.warning(
        "No data available for Diabetes Status distribution with the current filters."
    )
st.markdown("---")

# --- Categorical Feature Analysis Section ---
st.header("Categorical Features vs. Diabetes")
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
        "Select Categorical Feature",
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
                "Unknown": 6,
            }
        elif cat_feature_to_plot == "BMI_Category":
            category_orders = {
                "Underweight": 0,
                "Normal": 1,
                "Overweight": 2,
                "Obese": 3,
                "Unknown": 4,
            }
        elif cat_feature_to_plot == "BP_Category":
            category_orders = {
                "Normal": 0,
                "Elevated": 1,
                "Hypertension Stage 1": 2,
                "Hypertension Stage 2": 3,
                "Unknown": 4,
                "Undefined": 5,
            }
        elif cat_feature_to_plot == "FPG_Category":
            category_orders = {
                "Normal": 0,
                "Pre-diabetes (IFG)": 1,
                "Diabetes Threshold (>=7.0)": 2,
                "Unknown": 3,
            }

        present_cats = df[cat_feature_to_plot].unique()
        category_orders = {
            k: v for k, v in category_orders.items() if k in present_cats
        }

        fig_cat = px.histogram(
            df,
            x=cat_feature_to_plot,
            color="Diabetes_label",
            title=f"{cat_feature_to_plot.replace('_',' ').title()} Distribution by Diabetes Status",
            barmode="group",
            color_discrete_map=DIABETES_COLOR_MAP,  # Apply the defined Reds map
            category_orders=(
                {cat_feature_to_plot: sorted(category_orders, key=category_orders.get)}
                if category_orders
                else None
            ),
            labels={"Diabetes_label": "Diabetes Status"},
        )
        fig_cat.update_layout(
            xaxis_title=cat_feature_to_plot.replace("_", " ").title(),
            yaxis_title="Count",
        )
        st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.warning(
        "No categorical features available for distribution analysis with current filters."
    )
st.markdown("---")

# --- Diabetes Prevalence within Categories Section ---
st.header("Diabetes Prevalence within Categories")
available_cats_prev = [
    col
    for col in CATEGORICAL_COLS_MAPPED
    if col in df.columns and df[col].nunique() > 1
]

if available_cats_prev:
    cat_feature_for_prevalence = st.selectbox(
        "Select Category for Prevalence Analysis",
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
                "Unknown": 6,
            }
        elif cat_feature_for_prevalence == "BMI_Category":
            ordered_categories_prev = {
                "Underweight": 0,
                "Normal": 1,
                "Overweight": 2,
                "Obese": 3,
                "Unknown": 4,
            }
        elif cat_feature_for_prevalence == "BP_Category":
            ordered_categories_prev = {
                "Normal": 0,
                "Elevated": 1,
                "Hypertension Stage 1": 2,
                "Hypertension Stage 2": 3,
                "Unknown": 4,
                "Undefined": 5,
            }
        elif cat_feature_for_prevalence == "FPG_Category":
            ordered_categories_prev = {
                "Normal": 0,
                "Pre-diabetes (IFG)": 1,
                "Diabetes Threshold (>=7.0)": 2,
                "Unknown": 3,
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
            y="Prevalence (%)",
            title=f"Diabetes Prevalence within each '{cat_feature_for_prevalence.replace('_',' ').title()}' Category",
            color="Prevalence (%)",
            color_continuous_scale=REDS_COLOR_SCALE,  # Apply the Reds scale
            text_auto=".1f",
            labels={"Prevalence (%)": "Diabetes Prevalence (%)"},
        )
        # Ensure bar colors are set based on the continuous scale
        fig_prev.update_traces(marker_coloraxis="coloraxis")
        fig_prev.update_traces(textangle=0, textposition="outside")
        fig_prev.update_layout(
            xaxis_title=cat_feature_for_prevalence.replace("_", " ").title(),
            yaxis_title="Diabetes Prevalence (%)",
            yaxis_ticksuffix="%",
        )
        st.plotly_chart(fig_prev, use_container_width=True)

        with st.expander("Show Prevalence Data Table"):
            st.dataframe(
                prevalence_df[
                    [
                        cat_feature_for_prevalence,
                        "Diabetic Count",
                        "Total Count",
                        "Prevalence (%)",
                    ]
                ].round(1)
            )
else:
    st.warning(
        "No suitable categorical features available for prevalence analysis with current filters (need >1 category)."
    )
st.markdown("---")

# --- Numerical Feature Distribution Section ---
st.header("Numerical Feature Distributions & Clinical Thresholds")
numerical_cols = sorted(
    [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in ["Diabetes", "Gender", "smoking", "drinking", "family_history"]
        and df[col].nunique() > 1
    ]
)

if not numerical_cols:
    st.warning(
        "No numerical features with variance available for distribution analysis with current filters."
    )
else:
    default_feature_idx = 0
    if "FPG" in numerical_cols:
        default_feature_idx = numerical_cols.index("FPG")
    elif "BMI" in numerical_cols:
        default_feature_idx = numerical_cols.index("BMI")
    feature_to_plot = st.selectbox(
        "Select Numerical Feature to Visualize",
        numerical_cols,
        index=default_feature_idx,
        key="dist_feature_select",
    )

    if feature_to_plot and total_patients > 0:
        hist = dict(
            x=feature_to_plot,
            color="Diabetes_label",
            marginal="box",
            color_discrete_map=DIABETES_COLOR_MAP,  # Apply the defined Reds map
            barmode="overlay",
            opacity=0.7,
            labels={"Diabetes_label": "Diabetes Status"},
        )

        fig_dist_num = px.histogram(
            df,
            **hist,
            title=f"Histogram of {feature_to_plot} by Diabetes Status",
        )

        fig_dist_num.update_layout(xaxis_title=feature_to_plot)
        st.plotly_chart(fig_dist_num, use_container_width=True)

        fig_box = px.box(
            df,
            x="Diabetes_label",
            y=feature_to_plot,
            color="Diabetes_label",
            title=f"Box Plot of {feature_to_plot} by Diabetes Status",
            color_discrete_map=DIABETES_COLOR_MAP,  # Apply the defined Reds map
            labels={
                "Diabetes_label": "Diabetes Status",
                feature_to_plot: feature_to_plot,
            },
            points="outliers",
            notched=True,
        )
        st.plotly_chart(fig_box, use_container_width=True)
st.markdown("---")

# --- Bivariate Analysis Section (Scatter Plot) ---
st.header("Bivariate Analysis (Scatter Plot)")
if len(numerical_cols) < 2:
    st.warning(
        "Need at least two numerical features with variance for scatter plot analysis with current filters."
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
        "Select X-axis Feature", numerical_cols, index=default_x_idx, key="scatter_x"
    )
    y_axis = col_scatter2.selectbox(
        "Select Y-axis Feature", numerical_cols, index=default_y_idx, key="scatter_y"
    )

    if x_axis and y_axis and x_axis != y_axis and total_patients > 0:
        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color="Diabetes_label",
            title=f"{y_axis} vs. {x_axis} by Diabetes Status",
            color_discrete_map=DIABETES_COLOR_MAP,  # Apply the defined Reds map
            hover_data=["Age", "Gender_label", "BMI", "FPG", "SBP", "DBP"],
            labels={"Diabetes_label": "Diabetes Status"},
            opacity=0.7,
        )
        fig_scatter.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
        st.plotly_chart(fig_scatter, use_container_width=True)
    elif x_axis == y_axis:
        st.warning("Please select different features for the X and Y axes.")
st.markdown("---")

# --- Correlation Analysis Section ---
st.header("Correlation Matrix of Numerical Features")
corr_cols_available = [
    col for col in CORRELATION_COLS if col in df.columns and df[col].nunique() > 1
]

if len(corr_cols_available) > 1 and total_patients > 1:
    corr_subset = st.radio(
        "Calculate Correlation For:",
        ("All Filtered Data", "Diabetic Only", "Non-Diabetic Only"),
        horizontal=True,
        key="corr_radio",
        help="Calculate correlations on different subsets of the currently filtered data.",
    )
    df_corr_subset = df
    if corr_subset == "Diabetic Only":
        df_corr_subset = df[df["Diabetes"] == 1].copy()
    elif corr_subset == "Non-Diabetic Only":
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
            title=f"{corr_method.capitalize()} Correlation Matrix ({corr_subset})",
            color_continuous_scale=REDS_COLOR_SCALE,  # Apply the Reds scale
        )
        fig_corr.update_xaxes(side="bottom")
        fig_corr.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning(
            f"Not enough data (need > 1 row and > 1 column with variance) in the '{corr_subset}' group for correlation calculation with current filters."
        )
else:
    st.warning(
        "Not enough numerical columns with variance (need > 1) or data points (need > 1) for correlation matrix based on current filters."
    )
st.markdown("---")

# --- Line Graph Analysis Section (Trends across ordered categories) ---
st.header("Diabetes Prevalence Trends")
grouping_options_ordered = ["Age_Group", "BMI_Category", "BP_Category", "FPG_Category"]
available_groupings = [
    opt
    for opt in grouping_options_ordered
    if opt in df.columns and df[opt].nunique() > 1
]

if available_groupings:
    x_grouping = st.selectbox(
        "Group Prevalence By (X-axis)",
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
    available_segments = ["None"] + [
        opt
        for opt in segment_options_all
        if opt in df.columns and opt != x_grouping and df[opt].nunique() > 1
    ]
    segment_by = st.selectbox(
        "Segment Lines By (Optional)", available_segments, key="line_segment"
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
                "Unknown": 6,
            }
        elif x_grouping == "BMI_Category":
            ordered_categories = {
                "Underweight": 0,
                "Normal": 1,
                "Overweight": 2,
                "Obese": 3,
                "Unknown": 4,
            }
        elif x_grouping == "BP_Category":
            ordered_categories = {
                "Normal": 0,
                "Elevated": 1,
                "Hypertension Stage 1": 2,
                "Hypertension Stage 2": 3,
                "Unknown": 4,
                "Undefined": 5,
            }
        elif x_grouping == "FPG_Category":
            ordered_categories = {
                "Normal": 0,
                "Pre-diabetes (IFG)": 1,
                "Diabetes Threshold (>=7.0)": 2,
                "Unknown": 3,
            }

        grouping_cols = [x_grouping]
        if segment_by != "None":
            grouping_cols.append(segment_by)
        grouped_data = (
            df.groupby(grouping_cols).apply(safe_calculate_prevalence).reset_index()
        )
        grouped_data = grouped_data[grouped_data["Total Count"] > 0]

        if ordered_categories:
            present_cats_line = grouped_data[x_grouping].unique()
            current_order = {
                k: v for k, v in ordered_categories.items() if k in present_cats_line
            }
            grouped_data["sort_order"] = grouped_data[x_grouping].map(current_order)
            sort_cols = (
                [segment_by, "sort_order"] if segment_by != "None" else ["sort_order"]
            )
            grouped_data = grouped_data.sort_values(sort_cols).drop(
                "sort_order", axis=1
            )
        elif segment_by != "None":
            grouped_data = grouped_data.sort_values([segment_by, x_grouping])
        else:
            grouped_data = grouped_data.sort_values(x_grouping)

        plot_title = f"Diabetes Prevalence by {x_grouping.replace('_', ' ').title()}"
        color_col = segment_by if segment_by != "None" else None

        color_args = {}
        if segment_by == "None":
            color_args["color_discrete_sequence"] = [
                SINGLE_LINE_COLOR
            ]  
        else:
            plot_title += f" and {segment_by.replace('_', ' ').title()}"
            color_args["color"] = segment_by
            color_args["color_discrete_sequence"] = REDS_DISCRETE_SEQUENCE[3:]

        fig_line = px.line(
            grouped_data,
            x=x_grouping,
            y="Prevalence (%)",
            title=plot_title,
            markers=True,
            line_shape="linear",
            labels={"Prevalence (%)": "Diabetes Prevalence (%)"},
            hover_data=["Diabetic Count", "Total Count"],
            **color_args,  
        )

        fig_line.update_layout(
            xaxis_title=x_grouping.replace("_", " ").title(),
            yaxis_title="Diabetes Prevalence (%)",
            yaxis_ticksuffix="%",
            legend_title=(
                segment_by.replace("_", " ").title() if segment_by != "None" else None
            ),
        )

        if segment_by == "None":
            fig_line.update_traces(
                text=grouped_data["Prevalence (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="top center",
                mode="lines+markers+text",
                line=dict(width=2.5), 
            )
        else:
            fig_line.update_traces(mode="lines+markers", line=dict(width=3))

        st.plotly_chart(fig_line, use_container_width=True)

        with st.expander("Show Trend Data Table"):
            display_cols = grouping_cols + [
                "Diabetic Count",
                "Total Count",
                "Prevalence (%)",
            ]
            st.dataframe(grouped_data[display_cols].round(1))
else:
    st.warning(
        "No suitable ordered categorical variables available for line graph analysis with current filters."
    )
st.markdown("---")


# --- Data Explorer ---
st.header("Filtered Data Explorer")
st.markdown("View the raw data based on the filters applied (all columns shown).")
st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_diabetes_data.csv",
    mime="text/csv",
)
