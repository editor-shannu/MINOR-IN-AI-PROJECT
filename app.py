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


# -----------------------
# Offline AI intelligence
# -----------------------
def offline_reply(intent, df):
    latest = df.iloc[-1]

    if intent == "greeting":
        return "Hello! I can help analyze disease trends, market demand, and outbreak risks."

    if intent == "risk":
        return (
            f"The current risk score is {latest['risk_score']:.2f}. "
            "It is derived from disease growth, sales growth, public sentiment changes, "
            "and search trend spikes."
        )

    if intent == "forecast":
        return (
            "Forecasts are generated using time-series models like ARIMA, "
            "which capture historical trends and seasonality in disease cases."
        )

    if intent == "market":
        return (
            "Market behavior typically reacts to disease outbreaks with a time lag. "
            "Rising cases and public concern often lead to increased medicine demand."
        )

    return (
        "I can explain outbreak risks, forecasts, and market demand patterns. "
        "Try asking about risk, forecast, or market behavior."
    )


# -----------------------
# LLM-powered reply
# -----------------------
def llm_reply(question, df):
    import openai

    context = f"""
    Year: {df['year'].iloc[-1]}
    Total cases: {df['Total_cases'].iloc[-1]}
    Sales: {df['sales'].iloc[-1]}
    Risk score: {df['risk_score'].iloc[-1]:.2f}
    """

    prompt = f"""
    You are a healthcare market intelligence AI.
    Use the data below to answer analytically and clearly.

    DATA:
    {context}

    QUESTION:
    {question}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
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
    st.info("ℹ️ AI Assistant is running in offline intelligence mode. Add OPENAI_API_KEY for full LLM explanations.")

# =======================
# Dataset Preview
# =======================
with st.expander("📄 View Aggregated Dataset"):
    st.dataframe(yearly_df)

st.caption(
    "⚠️ For academic and decision-support use only. Predictions are based on historical data."
)