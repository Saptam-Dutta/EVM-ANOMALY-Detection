import streamlit as st
import pandas as pd
import os

from dotenv import load_dotenv
from google import genai

from anomaly_detection import calculate_evm, detect_anomalies


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="EVM Anomaly Detection",
    page_icon="📊",
    layout="wide"
)


# -----------------------------------
# Load Environment Variables
# -----------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# -----------------------------------
# Gemini Function
# -----------------------------------

def analyze_with_gemini(anomalies):

    if not GEMINI_API_KEY:
        return "Gemini API key not configured."

    client = genai.Client(api_key=GEMINI_API_KEY)

    data = anomalies[
        [
            "Sprint",
            "PV",
            "EV",
            "AC",
            "CV",
            "SV",
            "CPI",
            "SPI",
            "Status"
        ]
    ].to_string(index=False)

    prompt = f"""
You are an Earned Value Management expert.

Analyze the following EVM anomaly detection results:

{data}

For each detected anomaly:

1. Explain why it is anomalous.
2. Explain the cost variance (CV).
3. Explain the schedule variance (SV).
4. Interpret CPI and SPI.
5. Identify the main project risk.
6. Give a practical corrective action.

Keep the response concise and suitable for a college project report.
"""

    try:

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )

        return interaction.output_text

    except Exception as e:

        return f"Gemini API error: {str(e)}"

# -----------------------------------
# Title
# -----------------------------------

st.title("📊 EVM Anomaly Detection System")

st.write(
    "AI-assisted anomaly detection for Earned Value Management (EVM) data."
)


# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.header("Data Input")

uploaded_file = st.sidebar.file_uploader(
    "Upload EVM CSV file",
    type=["csv"]
)


# -----------------------------------
# Load Data
# -----------------------------------

if uploaded_file:

    df = pd.read_csv(uploaded_file)

else:

    try:
        df = pd.read_csv("evm_data.csv")

    except FileNotFoundError:

        st.error("evm_data.csv not found.")
        st.stop()


# -----------------------------------
# Validate Data
# -----------------------------------

required_columns = [
    "Sprint",
    "PV",
    "EV",
    "AC"
]

if not all(column in df.columns for column in required_columns):

    st.error(
        "CSV must contain: Sprint, PV, EV and AC"
    )

    st.stop()


# -----------------------------------
# EVM Calculations
# -----------------------------------

df = calculate_evm(df)


# -----------------------------------
# Anomaly Detection
# -----------------------------------

df = detect_anomalies(df)


# -----------------------------------
# Dashboard Metrics
# -----------------------------------

total_sprints = len(df)

anomaly_count = len(
    df[df["Status"] == "ANOMALY"]
)

normal_count = len(
    df[df["Status"] == "NORMAL"]
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Sprints",
        total_sprints
    )

with col2:
    st.metric(
        "Normal",
        normal_count
    )

with col3:
    st.metric(
        "Anomalies",
        anomaly_count
    )


# -----------------------------------
# EVM Data Table
# -----------------------------------

st.subheader("EVM Analysis")

st.dataframe(
    df,
    use_container_width=True
)


# -----------------------------------
# Anomaly Table
# -----------------------------------

st.subheader("🚨 Detected Anomalies")

anomalies = df[
    df["Status"] == "ANOMALY"
]

if len(anomalies) == 0:

    st.success("No anomalies detected.")

else:

    st.dataframe(
        anomalies,
        use_container_width=True
    )


# -----------------------------------
# Charts
# -----------------------------------

st.subheader("CPI Trend")

st.line_chart(
    df.set_index("Sprint")["CPI"]
)


st.subheader("SPI Trend")

st.line_chart(
    df.set_index("Sprint")["SPI"]
)


st.subheader("Cost Variance")

st.bar_chart(
    df.set_index("Sprint")["CV"]
)


st.subheader("Schedule Variance")

st.bar_chart(
    df.set_index("Sprint")["SV"]
)


# -----------------------------------
# Gemini Analysis
# -----------------------------------

st.subheader("🤖 Gemini AI Risk Analysis")

if st.button("Analyze Anomalies with Gemini"):

    if len(anomalies) == 0:

        st.info("No anomalies available for Gemini analysis.")

    else:

        with st.spinner("Gemini is analyzing the detected risks..."):

            analysis = analyze_with_gemini(
                anomalies
            )

        st.markdown(analysis)