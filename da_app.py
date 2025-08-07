import streamlit as st
import pandas as pd
from utils.eda_utils import generate_eda_report
from utils.stats_utils import simulate_confidence_intervals

st.set_page_config(layout="wide")
st.title("📊 Interactive EDA and Statistical Analysis App")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 📌 Data Preview")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("🧪 Choose Action")

    action = st.selectbox("Select analysis type", [
        "Generate EDA Report",
        "Simulate Confidence Intervals"
    ])

    if action == "Generate EDA Report":
        
        with open("eda_report.pdf", "rb") as f:
            
            
            st.download_button(
            label="📥 Download EDA Report",
            data=f,
            file_name="eda_report.pdf",
            mime="application/pdf"    
    )

        if st.button("Generate PDF Report"):
            generate_eda_report(df)
            st.success("✅ Report saved as `eda_report.pdf`")

    elif action == "Simulate Confidence Intervals":
        confidence = st.slider("Select Confidence Level (%)", 80, 99, 95)
        sample_size = st.slider("Sample Size", min_value=10, max_value=100, value=30)
        simulate_confidence_intervals(df, confidence, sample_size)

