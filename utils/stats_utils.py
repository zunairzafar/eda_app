import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def simulate_confidence_intervals(df, confidence, sample_size):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_col = st.selectbox("Choose numerical column for CI simulation", numeric_cols)
    if selected_col:
        data = df[selected_col].dropna()
        mean = data.mean()
        std = data.std()
        dof = sample_size - 1
        t_crit = t.ppf((1 + confidence / 100) / 2, df=dof)
        standard_error = std / (sample_size ** 0.5)
        lower = mean - t_crit * standard_error
        upper = mean + t_crit * standard_error

        st.markdown(f"**Confidence Interval ({confidence}%):** ({lower:.2f}, {upper:.2f})")
        st.markdown(f"**Sample Mean:** {mean:.2f}")

        # Plot CI distribution
        simulations = []
        for _ in range(100):
            sample = data.sample(sample_size, replace=True)
            sample_mean = sample.mean()
            se = sample.std() / (sample_size ** 0.5)
            margin = t_crit * se
            ci_low = sample_mean - margin
            ci_high = sample_mean + margin
            simulations.append((ci_low, ci_high, sample_mean))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (low, high, sm) in enumerate(simulations):
            color = "green" if low <= mean <= high else "red"
            ax.plot([low, high], [i, i], color=color)
            ax.plot(sm, i, "o", color="black")
        ax.axvline(mean, color="blue", linestyle="--", label="True Mean")
        ax.set_title(f"{confidence}% Confidence Intervals for {selected_col}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Simulation Index")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
