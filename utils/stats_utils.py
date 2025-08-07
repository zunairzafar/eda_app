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

     from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# Save the CI plot to an image buffer
img_buf = BytesIO()
fig.savefig(img_buf, format='png', bbox_inches='tight')
img_buf.seek(0)

# Generate a PDF and embed the image
pdf_buf = BytesIO()
c = canvas.Canvas(pdf_buf, pagesize=letter)
c.setFont("Helvetica-Bold", 14)
c.drawString(50, 750, f"{confidence}% Confidence Interval Report for {selected_col}")

# Embed the plot image
image = ImageReader(img_buf)
c.drawImage(image, 50, 400, width=500, height=300, preserveAspectRatio=True)

# Add interval details
c.setFont("Helvetica", 12)
c.drawString(50, 370, f"Sample Size: {sample_size}")
c.drawString(50, 350, f"Sample Mean: {mean:.2f}")
c.drawString(50, 330, f"Confidence Interval: ({lower:.2f}, {upper:.2f})")

c.showPage()
c.save()
pdf_buf.seek(0)

# Add download button
st.download_button(
    label="📥 Download CI Report as PDF",
    data=pdf_buf,
    file_name=f"ci_report_{selected_col}.pdf",
    mime="application/pdf"
)

        )
