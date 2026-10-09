
import streamlit as st
import pandas as pd
from google import genai
from openai import OpenAI
from anomaly_detection import calculate_evm, detect_anomalies


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EVM Anomaly Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Consistent light styling
st.markdown("""
<style>
    :root {
        color-scheme: light;
    }

    .stApp {
        background: #f4f7fb;
        color: #172b4d;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.96);
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] * {
        color: #24364b;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #ffffff, #eaf1ff);
        border: 1px solid #dce6f5;
        border-radius: 18px;
        padding: 30px 32px;
        margin-bottom: 26px;
    }

    .hero h1 {
        color: #172b4d;
        font-size: 2.2rem;
        font-weight: 750;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #5d6e85;
        font-size: 1rem;
        margin: 0;
    }

    h2, h3 {
        color: #172b4d !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(31, 55, 90, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #172b4d;
    }

    div[data-testid="stDataFrame"] {
        background: #ffffff;
        border: 1px solid #e1e8f0;
        border-radius: 12px;
        overflow: hidden;
    }

    .stButton > button {
        background: #2563eb;
        color: #ffffff;
        border: 1px solid #2563eb;
        border-radius: 9px;
        min-height: 44px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        border-color: #1d4ed8;
        color: #ffffff;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #e1e8f0;
    }

    .report {
        background: #ffffff;
        border: 1px solid #e1e8f0;
        border-radius: 14px;
        padding: 24px;
    }

    .muted {
        color: #64748b;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# AI REPORT GENERATION
# =========================================================

def generate_ai_report(provider, api_key, model, anomalies, base_url=""):
    if not api_key.strip():
        raise ValueError("Please enter an API key.")

    if not model.strip():
        raise ValueError("Please enter a model ID.")

    if anomalies.empty:
        raise ValueError("There are no flagged anomalies to analyze.")

    columns = [
        "Sprint", "PV", "EV", "AC",
        "CV", "SV", "CPI", "SPI", "Status"
    ]

    data = anomalies[columns].to_string(index=False)

    prompt = f"""
You are an Earned Value Management analyst.

Analyze these EVM records flagged by Isolation Forest:

{data}

Write a professional report containing:
1. Executive summary.
2. Cost and schedule variance analysis.
3. CPI and SPI interpretation.
4. Main project risks.
5. Recommended corrective actions.

Use only the supplied data. Explain that a statistical anomaly
is a signal to investigate, not proof of fraud or error.
Keep the report concise and suitable for a college assignment.
"""

    if provider == "Google Gemini":
        # Gemini models requiring Interactions API must use this
        # endpoint instead of models.generate_content().
        client = genai.Client(api_key=api_key.strip())

        response = client.interactions.create(
            model=model.strip(),
            input=prompt,
        )

        result = response.output_text

    else:
        # OpenAI and providers implementing compatible chat completions.
        client = OpenAI(
            api_key=api_key.strip(),
            base_url=base_url.strip() or None,
            timeout=60.0,
            max_retries=0,
        )

        response = client.chat.completions.create(
            model=model.strip(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful Earned Value Management analyst."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        result = response.choices[0].message.content

    if not result:
        raise ValueError("The model returned an empty response.")

    return result


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Configuration")
st.sidebar.caption("Connect an AI provider to generate a report.")

provider = st.sidebar.selectbox(
    "AI provider",
    ["Google Gemini", "OpenAI-compatible API"],
)

api_key = st.sidebar.text_input(
    "API key",
    type="password",
    placeholder="Paste your API key",
    help="Your key is used for the API request.",
)

base_url = ""

if provider == "OpenAI-compatible API":
    base_url = st.sidebar.text_input(
        "API base URL",
        placeholder="https://api.openai.com/v1",
        help=(
            "Enter the provider's documented OpenAI-compatible "
            "base URL. Leave blank for OpenAI."
        ),
    )

if provider == "Google Gemini":
    model = st.sidebar.text_input(
        "Model ID",
        value="gemini-3.8-flash",
        help="Use a model ID available to your Google AI API key.",
    )
else:
    model = st.sidebar.text_input(
        "Model ID",
        value="",
        placeholder="Enter your provider's model ID",
        help="Use a model ID supported by the configured endpoint.",
    )

st.sidebar.divider()
st.sidebar.subheader("Project data")

uploaded_file = st.sidebar.file_uploader(
    "Upload EVM CSV",
    type=["csv"],
    help="Required columns: Sprint, PV, EV, AC",
)

st.sidebar.caption(
    "API keys from different providers are not interchangeable. "
    "This app does not save keys to a file."
)


# =========================================================
# PAGE HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>EVM Anomaly Detection</h1>
    <p>
        Earned Value Management analytics, statistical anomaly
        detection, and AI-assisted project risk reporting.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("evm_data.csv")
except FileNotFoundError:
    st.error("evm_data.csv was not found. Upload a CSV to continue.")
    st.stop()
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

required = ["Sprint", "PV", "EV", "AC"]
missing = [column for column in required if column not in df.columns]

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

if df.empty:
    st.error("The CSV file contains no data rows.")
    st.stop()

for column in ["PV", "EV", "AC"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if df[["PV", "EV", "AC"]].isna().any().any():
    st.error("PV, EV and AC must contain valid numeric values.")
    st.stop()

if (df["PV"] <= 0).any() or (df["AC"] <= 0).any():
    st.error("PV and AC must be greater than zero.")
    st.stop()

try:
    df = calculate_evm(df)
    df = detect_anomalies(df)
except Exception as exc:
    st.error(f"Anomaly detection failed: {exc}")
    st.stop()

anomalies = df[df["Status"] == "ANOMALY"].copy()
normal_count = int((df["Status"] == "NORMAL").sum())


# =========================================================
# OVERVIEW
# =========================================================

st.subheader("Project overview")

m1, m2, m3 = st.columns(3)

m1.metric("Total sprints", len(df))
m2.metric("Normal records", normal_count)
m3.metric("Flagged anomalies", len(anomalies))

st.write("")


# =========================================================
# EVM RESULTS
# =========================================================

st.subheader("Earned Value Management results")

display_columns = [
    "Sprint", "PV", "EV", "AC",
    "CV", "SV", "CPI", "SPI", "Status"
]

st.dataframe(
    df[display_columns].round(
        {"CPI": 3, "SPI": 3}
    ),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Anomaly review")

if anomalies.empty:
    st.info("No records were flagged by Isolation Forest.")
else:
    st.dataframe(
        anomalies[display_columns].round(
            {"CPI": 3, "SPI": 3}
        ),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "An anomaly is a statistical flag that requires investigation; "
    "it does not automatically establish that the data is wrong."
)


# =========================================================
# CHARTS
# =========================================================

st.subheader("Performance trends")

left, right = st.columns(2)

with left:
    st.markdown("**Cost Performance Index (CPI)**")
    st.line_chart(df.set_index("Sprint")[["CPI"]])

with right:
    st.markdown("**Schedule Performance Index (SPI)**")
    st.line_chart(df.set_index("Sprint")[["SPI"]])

left, right = st.columns(2)

with left:
    st.markdown("**Cost variance**")
    st.bar_chart(df.set_index("Sprint")[["CV"]])

with right:
    st.markdown("**Schedule variance**")
    st.bar_chart(df.set_index("Sprint")[["SV"]])


# =========================================================
# AI REPORT
# =========================================================

st.divider()
st.subheader("AI risk analysis")

st.write(
    "The selected AI model will explain the detected anomalies "
    "and recommend corrective actions."
)

if st.button("Generate AI report", type="primary", use_container_width=True):
    if not api_key.strip():
        st.warning("Enter your API key in the sidebar.")
    elif not model.strip():
        st.warning("Enter a model ID in the sidebar.")
    elif provider == "OpenAI-compatible API" and base_url.strip() and not base_url.startswith(("https://", "http://")):
        st.warning("The API base URL must start with http:// or https://.")
    elif anomalies.empty:
        st.info("There are no flagged anomalies to send for analysis.")
    else:
        try:
            with st.spinner("Generating your risk report..."):
                report = generate_ai_report(
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    anomalies=anomalies,
                    base_url=base_url,
                )

            st.markdown('<div class="report">', unsafe_allow_html=True)
            st.markdown("### Generated risk report")
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as exc:
            error_text = str(exc)
            st.error("The AI request could not be completed.")

            if "401" in error_text or "authentication" in error_text.lower():
                st.warning("Check that the API key is valid for this provider.")
            elif "404" in error_text or "not found" in error_text.lower():
                st.warning(
                    "Check the model ID and whether your API key has access to it."
                )
            elif "429" in error_text or "quota" in error_text.lower():
                st.warning(
                    "The provider may have rate-limited the request or "
                    "your quota may be exhausted."
                )
            elif "400" in error_text and provider == "Google Gemini":
                st.warning(
                    "Check that the model supports the Gemini Interactions API "
                    "and that the selected model ID is correct."
                )

            with st.expander("Technical error details"):
                st.code(error_text)