import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from scipy.stats import norm
import os

def generate_eda_report(df):
    doc = SimpleDocTemplate("eda_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Exploratory Data Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = [col for col in df.columns if df[col].nunique() <= 10]

    # Descriptive stats
    elements.append(Paragraph("Descriptive Statistics", styles['Heading2']))
    desc_stats = df.describe().round(2).to_string()
    elements.append(Paragraph(f"<pre>{desc_stats}</pre>", styles['Code']))
    elements.append(Spacer(1, 12))

    # Distribution and CLT
    for col in numeric_cols:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=20)
        ax.set_title(f"Distribution of {col}")
        img_path = f"{col}_hist.png"
        plt.savefig(img_path)
        elements.append(Image(img_path, width=400, height=300))
        plt.close()

        # CLT simulation
        sample_means = [df[col].dropna().sample(30).mean() for _ in range(1000)]
        fig, ax = plt.subplots()
        ax.hist(sample_means, bins=30)
        ax.set_title(f"CLT Simulation for {col}")
        img_path = f"{col}_clt.png"
        plt.savefig(img_path)
        elements.append(Image(img_path, width=400, height=300))
        plt.close()

    # Boxplots and outliers
    for col in numeric_cols:
        fig, ax = plt.subplots()
        df.boxplot(column=col, ax=ax)
        ax.set_title(f"Boxplot of {col}")
        img_path = f"{col}_box.png"
        plt.savefig(img_path)
        elements.append(Image(img_path, width=400, height=300))
        plt.close()

    doc.build(elements)

    # Clean up
    for col in numeric_cols:
        for suffix in ["_hist.png", "_clt.png", "_box.png"]:
            try:
                os.remove(f"{col}{suffix}")
            except:
                pass
