import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =======================
# Page Config
# =======================
st.set_page_config(
    page_title="Disease & Medicine Sales Dashboard",
    layout="wide"
)

# =======================
# Load Data
# =======================
@st.cache_data
def load_data():
    return pd.read_csv("combined.csv")

combined = load_data()

# =======================
# Safe Aggregation
# =======================
sentiment_col = "bert_sentiment_score"

agg_dict = {
    "Total_cases": "mean",
    "sales": "mean",
    "flu_trend": "mean"
}

if sentiment_col in combined.columns:
    agg_dict[sentiment_col] = "mean"

yearly_df = (
    combined
    .groupby("year", as_index=False)
    .agg(agg_dict)
)

# =======================
# Health Index (Multi-signal Fusion)
# =======================
if sentiment_col in yearly_df.columns:
    yearly_df["health_index"] = (
        0.5 * yearly_df["Total_cases"]
        + 0.3 * yearly_df["flu_trend"]
        + 0.2 * yearly_df[sentiment_col]
    )
else:
    yearly_df["health_index"] = (
        0.6 * yearly_df["Total_cases"]
        + 0.4 * yearly_df["flu_trend"]
    )

# =======================
# REAL SIGNAL ENGINEERING
# =======================

yearly_df["case_growth"] = yearly_df["Total_cases"].pct_change()
yearly_df["sales_growth"] = yearly_df["sales"].pct_change()

yearly_df["sentiment_change"] = (
    yearly_df[sentiment_col].diff()
    if sentiment_col in yearly_df.columns else 0
)

yearly_df["search_spike"] = yearly_df["flu_trend"].pct_change()

def safe_norm(series):
    return (series - series.mean()) / (series.std() + 1e-6)

yearly_df["risk_score"] = (
    0.35 * safe_norm(yearly_df["case_growth"]) +
    0.25 * safe_norm(yearly_df["sales_growth"]) +
    0.20 * safe_norm(yearly_df["sentiment_change"]) +
    0.20 * safe_norm(yearly_df["search_spike"])
)

# =======================
# Title
# =======================
st.title("🩺 Disease & Medicine Sales Forecasting Dashboard")

# =======================
# 🚨 EARLY WARNING BADGE
# =======================
st.subheader("⚠️ AI Early Warning System (Multi-Signal)")

latest_risk = yearly_df["risk_score"].iloc[-1]

# =======================
# HUMAN READABLE SIGNALS
# =======================

def risk_label(score):
    if score > 1.0:
        return "HIGH", "🔴"
    elif score > 0.3:
        return "MODERATE", "🟠"
    else:
        return "LOW", "🟢"

RISK_TEXT, RISK_ICON = risk_label(latest_risk)

if latest_risk > 1.0:
    st.error("🔴 HIGH RISK: Strong outbreak signals detected. Immediate demand surge likely.")
elif latest_risk > 0.3:
    st.warning("🟠 MODERATE RISK: Early warning signals rising. Monitor inventory closely.")
else:
    st.success("🟢 LOW RISK: Disease and market conditions stable.")

# =======================
# SECTION 1 — Overview
# =======================
st.subheader("📊 Disease Cases vs Medicine Sales")

fig1 = px.line(
    yearly_df,
    x="year",
    y=["Total_cases", "sales"],
    markers=True
)
st.plotly_chart(fig1, use_container_width=True)

if st.button("📊 Explain this chart"):
    explanation = (
        f"{RISK_ICON} **Risk Level: {RISK_TEXT}**\n\n"
        "This chart compares disease cases and medicine sales over time. "
        "Increases in disease cases are usually followed by higher medicine sales, "
        "indicating a lag-based demand response in the healthcare market."
    )
    st.session_state.chat.append(("AI", explanation))


# =======================
# SECTION 1A — Relationship
# =======================
st.subheader("🔗 Disease ↔ Medicine Sales Relationship")

corr = combined["Total_cases"].corr(combined["sales"])
lag_corr = combined["Total_cases"].shift(1).corr(combined["sales"])

st.write(f"**Correlation:** {corr:.2f}")
st.write(f"**Lag Correlation (Cases → Sales):** {lag_corr:.2f}")

# =======================
# SECTION 2 — Sentiment
# =======================
if sentiment_col in yearly_df.columns:
    st.subheader("🧠 Public Sentiment Trend")
    st.line_chart(yearly_df.set_index("year")[sentiment_col])

# =======================
# SECTION 3 — Google Trends
# =======================
st.subheader("🌐 Flu Search Interest")
st.line_chart(yearly_df.set_index("year")["flu_trend"])

# =======================
# SECTION — Health Index
# =======================
st.subheader("📈 Health Index vs Sales")
st.line_chart(yearly_df.set_index("year")[["health_index", "sales"]])

# =======================
# SECTION 4 — Moving Average
# =======================
st.subheader("📉 Moving Average Baseline")

yearly_df["MA_3"] = yearly_df["Total_cases"].rolling(3).mean()
st.line_chart(yearly_df.set_index("year")[["Total_cases", "MA_3"]])

ma_df = yearly_df.dropna(subset=["MA_3"])

if len(ma_df) > 1:
    mae_ma = mean_absolute_error(ma_df["Total_cases"], ma_df["MA_3"])
    rmse_ma = np.sqrt(mean_squared_error(ma_df["Total_cases"], ma_df["MA_3"]))
else:
    mae_ma, rmse_ma = 0, 0

# =======================
# SECTION 5 — ARIMA
# =======================
st.subheader("📈 ARIMA Forecast")

model = ARIMA(yearly_df["Total_cases"], order=(2,1,1))
res = model.fit()

forecast = res.forecast(steps=5)
future_years = [yearly_df["year"].max() + i for i in range(1,6)]

arima_df = pd.DataFrame({
    "year": future_years,
    "forecast": forecast
})

st.line_chart(
    pd.concat([
        yearly_df[["year","Total_cases"]].rename(columns={"Total_cases":"forecast"}),
        arima_df
    ]).set_index("year")
)

rmse_arima = np.sqrt(mean_squared_error(
    yearly_df["Total_cases"].iloc[-3:],
    res.predict(start=len(yearly_df)-3, end=len(yearly_df)-1)
))
mae_arima = mean_absolute_error(
    yearly_df["Total_cases"].iloc[-3:],
    res.predict(start=len(yearly_df)-3, end=len(yearly_df)-1)
)

forecast_volatility = np.std(res.resid)

confidence = max(0, 100 - forecast_volatility / yearly_df["Total_cases"].mean() * 100)

CONFIDENCE_TEXT = (
    "High confidence forecast"
    if confidence >= 70
    else "Moderate confidence forecast"
    if confidence >= 40
    else "Low confidence forecast"
)


st.metric(
    "📊 Forecast Confidence",
    f"{confidence:.1f}%",
    help="Derived from forecast residual volatility"
)

# =======================
# SECTION 6 — Model Metrics
# =======================
st.subheader("📐 Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "MA RMSE",
    "N/A" if np.isnan(rmse_ma) else f"{rmse_ma:.0f}"
)

c2.metric(
    "MA MAE",
    "N/A" if np.isnan(mae_ma) else f"{mae_ma:.0f}"
)
c3.metric("ARIMA RMSE", f"{rmse_arima:.0f}")
c4.metric("ARIMA MAE", f"{mae_arima:.0f}")
# =======================
# 🤖 AI ASSISTANT (DATA-AWARE)
# =======================
st.subheader("🧠 AI Market Intelligence Assistant")

# -----------------------
# One-time setup (cached)
# -----------------------
@st.cache_resource
def check_openai():
    try:
        import openai
        if "OPENAI_API_KEY" in st.secrets:
            openai.api_key = st.secrets["OPENAI_API_KEY"]
            return True
    except Exception:
        pass
    return False

OPENAI_AVAILABLE = check_openai()

# -----------------------
# Smart intent detection
# -----------------------
def detect_intent(text):
    text = text.lower()
    if any(k in text for k in ["hi", "hello", "hey"]):
        return "greeting"
    if "risk" in text:
        return "risk"
    if "forecast" in text or "predict" in text:
        return "forecast"
    if "market" in text or "behavior" in text or "demand" in text:
        return "market"
    return "general"

def offline_reply(intent, df):
    latest = df.iloc[-1]

    header = f"{RISK_ICON} **Current Risk Level: {RISK_TEXT}**\n\n"

    if intent == "greeting":
        return header + (
            "Hello. I analyze disease outbreaks, medicine demand, and "
            "healthcare market risks using real-time statistical signals."
        )

    if intent == "risk":
        return header + (
            f"The outbreak risk is **{RISK_TEXT}**. "
            f"Recent case growth is {latest['case_growth']:.2%}, "
            f"sales growth is {latest['sales_growth']:.2%}, "
            "indicating pressure on medicine demand."
        )

    if intent == "forecast":
        return header + (
            f"{CONFIDENCE_TEXT}. "
            "The forecast is driven by historical volatility and trend persistence "
            "in disease case data."
        )

    if intent == "market":
        return header + (
            "Market behavior shows a lagged response. "
            "Disease case increases are followed by higher medicine sales "
            "due to precautionary buying and treatment demand."
        )

    return header + (
        "I can explain current risks, forecast confidence, "
        "and healthcare market behavior. Try asking about risk or forecast."
    )

# -----------------------
# LLM-powered reply
# -----------------------
def llm_reply(question, df):
    import openai

    prompt = f"""
You are a healthcare market intelligence AI.

Current risk level: {RISK_TEXT}
Forecast confidence: {CONFIDENCE_TEXT}

DATA CONTEXT:
Year: {df['year'].iloc[-1]}
Total cases: {df['Total_cases'].iloc[-1]}
Medicine sales: {df['sales'].iloc[-1]}
Risk score: {df['risk_score'].iloc[-1]:.2f}

QUESTION:
{question}

Respond like a professional market analyst.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25
    )

    return response.choices[0].message.content



# -----------------------
# Chat state
# -----------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask about risks, forecasts, or market behavior")

if user_input:
    intent = detect_intent(user_input)

    if OPENAI_AVAILABLE:
        reply = llm_reply(user_input, yearly_df)
    else:
        reply = offline_reply(intent, yearly_df)

    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("AI", reply))

# Render chat
for speaker, msg in st.session_state.chat:
    st.markdown(f"**{speaker}:** {msg}")

# Single info banner (no repetition)
if not OPENAI_AVAILABLE:
    st.caption(
    "🧠 AI Assistant is running in **offline intelligence mode**, "
    "using real data signals. Connect an API key to enable deep LLM explanations."
)

# =======================
# Dataset Preview
# =======================
with st.expander("📄 View Aggregated Dataset"):
    st.dataframe(yearly_df)

st.caption(
    "⚠️ For academic and decision-support use only. Predictions are based on historical data."
)