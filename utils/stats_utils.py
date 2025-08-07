import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader


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

        # Simulate CI samples
        simulations = []
        for _ in range(100):
            sample = data.sample(sample_size, replace=True)
            sample_mean = sample.mean()
            se = sample.std() / (sample_size ** 0.5)
            margin = t_crit * se
            ci_low = sample_mean - margin
            ci_high = sample_mean + margin
            simulations.append((ci_low, ci_high, sample_mean))

        # Adjust figure height dynamically to fit screen
        fig_height = min(12, 0.1 * len(simulations))  # max 12 inches
        fig, ax = plt.subplots(figsize=(8, fig_height))

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

        # ====== Generate downloadable PDF ======
        # Save figure to buffer
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=300)
        img_buffer.seek(0)

        # Create PDF with image embedded
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, 770, f"CI Simulation Report - {selected_col} ({confidence}%)")

        # Draw the image on PDF
        image = ImageReader(img_buffer)
        c.drawImage(image, 40, 300, width=500, height=400, preserveAspectRatio=True)

        c.save()
        pdf_buffer.seek(0)

        # ====== Streamlit download button ======
        st.download_button(
            label="📥 Download CI Report as PDF",
            data=pdf_buffer,
            file_name=f"ci_report_{selected_col}_{confidence}pct.pdf",
            mime="application/pdf"
        )
