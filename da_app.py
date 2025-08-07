# streamlit_eda_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, norm
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import os

st.set_page_config(layout="wide")
st.title("📊 Interactive EDA & Statistical Analysis App")

# --- Upload file ---
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Loaded Successfully!")
    st.write(df.head())

    # --- Determine column types ---
    numerical_cols = []
    categorical_cols = []

    for col in df.columns:
        unique_vals = df[col].nunique()
        if pd.api.types.is_numeric_dtype(df[col]) and unique_vals > 10:
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    st.subheader("📌 Column Classification")
    st.write("**Numerical Columns:**", numerical_cols)
    st.write("**Categorical Columns:**", categorical_cols)

    # --- Confidence Interval Input ---
    st.subheader("📈 Confidence Interval Simulation")
    ci_col = st.selectbox("Select a numerical column for CI simulation", numerical_cols)
    ci_percent = st.slider("Confidence Level (%)", 80, 99, 95)

    ci_data = df[ci_col].dropna()
    sample_size = min(30, len(ci_data))
    sample = ci_data.sample(sample_size)

    sample_mean = sample.mean()
    sample_std = sample.std(ddof=1)
    dof = sample_size - 1
    confidence = ci_percent / 100
    t_crit = t.ppf((1 + confidence) / 2, dof)
    margin_error = t_crit * (sample_std / np.sqrt(sample_size))

    lower_bound = sample_mean - margin_error
    upper_bound = sample_mean + margin_error

    st.write(f"**Sample Size:** {sample_size}")
    st.write(f"**Sample Mean:** {sample_mean:.2f}")
    st.write(f"**Confidence Interval:** {lower_bound:.2f} to {upper_bound:.2f}")

    # --- Plot CI Simulation ---
    fig, ax = plt.subplots()
    ax.errorbar(1, sample_mean, yerr=margin_error, fmt='o', capsize=5)
    ax.axhline(sample_mean, color='green', linestyle='--')
    ax.set_xlim(0, 2)
    ax.set_title(f"{ci_percent}% CI for {ci_col}")
    st.pyplot(fig)

    # --- Distribution + CLT Analysis ---
    st.subheader("📊 Distribution Analysis & CLT")
    dist_col = st.selectbox("Choose column for distribution analysis", numerical_cols)

    fig1, ax1 = plt.subplots()
    sns.histplot(df[dist_col].dropna(), kde=True, ax=ax1)
    ax1.set_title(f"Distribution of {dist_col}")
    st.pyplot(fig1)

    skewness = df[dist_col].dropna().skew()
    st.write(f"**Skewness:** {skewness:.2f}")
    if abs(skewness) > 0.5:
        st.warning("Distribution is not normal — applying Central Limit Theorem (CLT)...")
        clt_means = [df[dist_col].dropna().sample(30).mean() for _ in range(1000)]
        fig2, ax2 = plt.subplots()
        sns.histplot(clt_means, kde=True, ax=ax2)
        ax2.set_title("Sampling Distribution of Mean (CLT)")
        st.pyplot(fig2)

    # --- Boxplots for Outliers ---
    st.subheader("📦 Boxplot & Outlier Detection")
    box_col = st.selectbox("Select column for boxplot", numerical_cols)
    fig3, ax3 = plt.subplots()
    sns.boxplot(df[box_col], ax=ax3)
    ax3.set_title(f"Boxplot of {box_col}")
    st.pyplot(fig3)

    q1 = df[box_col].quantile(0.25)
    q3 = df[box_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[box_col] < lower) | (df[box_col] > upper)]
    st.write(f"**Number of Outliers in {box_col}:** {len(outliers)}")

    # --- Generate PDF Report ---
    if st.button("📄 Generate PDF Report"):
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(tmpfile.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [Paragraph("Exploratory Data Analysis Report", styles["Title"]), Spacer(1, 12)]

        story.append(Paragraph("Numerical Columns", styles["Heading2"]))
        story.append(Paragraph(", ".join(numerical_cols), styles["Normal"]))
        story.append(Paragraph("Categorical Columns", styles["Heading2"]))
        story.append(Paragraph(", ".join(categorical_cols), styles["Normal"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph(f"{ci_percent}% Confidence Interval for {ci_col}: [{lower_bound:.2f}, {upper_bound:.2f}]", styles["Normal"]))
        story.append(Paragraph(f"Skewness of {dist_col}: {skewness:.2f}", styles["Normal"]))
        story.append(Paragraph(f"Outliers in {box_col}: {len(outliers)}", styles["Normal"]))

        doc.build(story)
        st.success("PDF Report Generated!")
        with open(tmpfile.name, "rb") as f:
            st.download_button("📥 Download Report", f, file_name="eda_report.pdf")
