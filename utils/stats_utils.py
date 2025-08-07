import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

def simulate_confidence_intervals(df, confidence, sample_size):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Choose numerical column", numeric_cols)

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

        simulations = []
        for _ in range(100):
            sample = data.sample(sample_size, replace=True)
            sample_mean = sample.mean()
            se = sample.std() / (sample_size ** 0.5)
            margin = t_crit * se
            ci_low = sample_mean - margin
            ci_high = sample_mean + margin
            simulations.append((ci_low, ci_high, sample_mean))

        fig, ax = plt.subplots(figsize=(10, 6))
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

        # Save CI report
        img_path = f"{selected_col}_ci_plot.png"
        fig.savefig(img_path)
        plt.close(fig)

        pdf_path = "ci_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("Confidence Interval Report", styles["Title"]), Spacer(1, 12)]
        story.append(Paragraph(f"Column: {selected_col}", styles["Normal"]))
        story.append(Paragraph(f"Confidence Level: {confidence}%", styles["Normal"]))
        story.append(Paragraph(f"Sample Size: {sample_size}", styles["Normal"]))
        story.append(Paragraph(f"Mean: {mean:.2f}", styles["Normal"]))
        story.append(Paragraph(f"CI: ({lower:.2f}, {upper:.2f})", styles["Normal"]))
        story.append(Spacer(1, 12))
        story.append(Image(img_path, width=400, height=300))
        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download CI Report", f, file_name="ci_report.pdf")
