import streamlit as st
import pandas as pd

# Load prepared datasets
combined = pd.read_csv("combined.csv")
sentiment = pd.read_csv("sentiment_yearly.csv")

# Merge (safe merge)
dashboard_df = combined.merge(sentiment, on="year", how="left")

st.title("Disease vs Medicine Sales Dashboard")

st.subheader("Disease Cases vs Medicine Sales")
st.line_chart(
    dashboard_df.set_index("year")[["Total_cases", "sales"]]
)

st.subheader("BERT Sentiment Trend")
st.line_chart(
    dashboard_df.set_index("year")[["bert_sentiment_score"]]
)

# Dropdown selection
metric = st.selectbox(
    "Select a metric to visualize",
    ["sales", "Total_cases", "bert_sentiment_score"]
)

st.subheader(f"Selected Metric: {metric}")
st.line_chart(
    dashboard_df.set_index("year")[[metric]]
)