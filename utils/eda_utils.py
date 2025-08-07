import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import seaborn as sns
import os

def generate_eda_report(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in df.columns if df[col].nunique() <= 10 or df[col].dtype == "object"]

    doc = SimpleDocTemplate("eda_report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("EDA Report", styles['Title']), Spacer(1, 12)]

    story.append(Paragraph("Descriptive Statistics:", styles["Heading2"]))
    desc = df.describe().round(2).to_string()
    story.append(Paragraph(f"<pre>{desc}</pre>", styles["Code"]))
    story.append(Spacer(1, 12))

    # Numerical Distribution
    story.append(Paragraph("Distribution of Numerical Columns:", styles["Heading2"]))
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
        img_path = f"{col}_dist.png"
        fig.savefig(img_path)
        plt.close(fig)
        story.append(Image(img_path, width=400, height=300))
        story.append(Spacer(1, 12))

        # CLT if not normal
        if abs(df[col].skew()) > 0.5:
            samples = [df[col].dropna().sample(30).mean() for _ in range(100)]
            fig, ax = plt.subplots()
            sns.histplot(samples, kde=True, ax=ax)
            ax.set_title(f"CLT Simulation (Mean Dist): {col}")
            clt_img = f"{col}_clt.png"
            fig.savefig(clt_img)
            plt.close(fig)
            story.append(Image(clt_img, width=400, height=300))
            story.append(Spacer(1, 12))

    # Boxplot and outliers
    story.append(Paragraph("Outlier Analysis with Boxplots:", styles["Heading2"]))
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot: {col}")
        img_path = f"{col}_box.png"
        fig.savefig(img_path)
        plt.close(fig)
        story.append(Image(img_path, width=400, height=300))
        story.append(Spacer(1, 12))

    # Categorical
    story.append(Paragraph("Categorical Feature Analysis:", styles["Heading2"]))
    for col in cat_cols:
        story.append(Paragraph(f"{col}: {df[col].value_counts().to_dict()}", styles["Normal"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    return "eda_report.pdf"

