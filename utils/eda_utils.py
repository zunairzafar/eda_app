# utils/eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def classify_columns(df, threshold=10):
    categorical_cols = []
    numerical_cols = []
    for col in df.columns:
        if df[col].dtype == 'O' or df[col].nunique() <= threshold:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    return categorical_cols, numerical_cols

def generate_missing_value_report(df):
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percent': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    return missing_df

def describe_data(df):
    return df.describe()

def plot_distribution(df, numerical_cols, output_dir):
    paths = []
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        path = os.path.join(output_dir, f"dist_{col}.png")
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths

def plot_boxplots(df, numerical_cols, output_dir):
    paths = []
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        path = os.path.join(output_dir, f"box_{col}.png")
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths

def identify_outliers(df, numerical_cols):
    outlier_report = {}
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_report[col] = len(outliers)
    return outlier_report
