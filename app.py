import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Anomaly & Risk Detection", layout="wide")

st.title("📊 Financial Anomaly & Risk Detection Dashboard")

np.random.seed(42)
N = 5000

df = pd.DataFrame({
    "transaction_amount": np.random.normal(200, 50, N).clip(10, 3000),
    "account_balance": np.random.normal(5000, 1500, N).clip(500, 25000),
    "transaction_hour": np.random.randint(0, 24, N),
    "is_international": np.random.choice([0, 1], N, p=[0.9, 0.1]),
    "merchant_risk_score": np.random.uniform(0, 1, N)
})

anomaly_idx = np.random.choice(N, int(0.02 * N), replace=False)
df.loc[anomaly_idx, "transaction_amount"] *= 5
df.loc[anomaly_idx, "merchant_risk_score"] = np.random.uniform(0.8, 1, len(anomaly_idx))

features = [
    "transaction_amount",
    "account_balance",
    "transaction_hour",
    "is_international",
    "merchant_risk_score"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

model = IsolationForest(contamination=0.02, random_state=42)
df["anomaly"] = model.fit_predict(X_scaled)
df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

df["risk_score"] = (
    df["merchant_risk_score"] * 0.5 +
    (df["transaction_amount"] / df["transaction_amount"].max()) * 0.5
)

total_tx = len(df)
anomalies = df[df["anomaly"] == "Anomaly"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_tx)
col2.metric("Detected Anomalies", len(anomalies))
col3.metric("High Risk Transactions", (df["risk_score"] > 0.7).sum())

st.subheader("Transaction Distribution")
fig, ax = plt.subplots()
ax.hist(df["transaction_amount"], bins=50)
st.pyplot(fig)

st.subheader("Anomalies Overview")
st.dataframe(anomalies.head(20))

st.subheader("🚨 Alerts")
high_risk = df[(df["anomaly"] == "Anomaly") & (df["risk_score"] > 0.7)]
st.write(high_risk.head(10))
