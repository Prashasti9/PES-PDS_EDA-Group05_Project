import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.set_page_config(page_title="NIFTY 50 EDA & Prediction", layout="wide")
st.title("📈 NIFTY 50 EDA & Market Movement Predictor")

# Upload CSV
df = None
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("NIFTY_50.csv")

# Preprocessing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(method='ffill', inplace=True)
df['Daily_Movement'] = np.where(df['Close'] > df['Open'], 'Up', 'Down')
df['Price_Range'] = df['High'] - df['Low']
df['Volatility_Level'] = pd.qcut(df['Price_Range'], q=3, labels=['Low', 'Medium', 'High'])

# Sidebar filter
st.sidebar.subheader("📅 Filter by Date")
start_date = st.sidebar.date_input("Start Date", df['Date'].min())
end_date = st.sidebar.date_input("End Date", df['Date'].max())
filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

# Summary
st.subheader("📌 Summary Statistics")
st.dataframe(filtered_df.describe())

# Line Chart
st.subheader("📉 Close Price Over Time")
st.line_chart(filtered_df.set_index('Date')['Close'])

# Correlation Heatmap
st.subheader("📊 Correlation Heatmap")
features = ['Open', 'High', 'Low', 'Close', 'Shares_Traded', 'Turnover_Rs_Cr']
fig_corr, ax_corr = plt.subplots()
sns.heatmap(filtered_df[features].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)

# Countplot for Daily Movement
st.subheader("📈 Market Movement Count")
fig_count, ax_count = plt.subplots()
sns.countplot(x='Daily_Movement', data=filtered_df, palette='Set2', ax=ax_count)
st.pyplot(fig_count)

# ML Prediction Inputs
st.subheader("🤖 Predict Daily Market Movement")
cols = st.columns(2)
with cols[0]:
    open_price = st.number_input("Open Price", value=18000)
    high_price = st.number_input("High Price", value=18200)
    low_price = st.number_input("Low Price", value=17900)
with cols[1]:
    shares_traded = st.number_input("Shares Traded", value=15000000)
    turnover = st.number_input("Turnover (Rs Cr)", value=20000.0)

# Model Training
X = df[['Open', 'High', 'Low', 'Shares_Traded', 'Turnover_Rs_Cr']]
y = df['Daily_Movement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📋 Model Evaluation")
st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
st.metric("Precision", f"{precision_score(y_test, y_pred, pos_label='Up'):.2f}")
st.metric("Recall", f"{recall_score(y_test, y_pred, pos_label='Up'):.2f}")

# Make Prediction
if st.button("🔮 Predict Movement"):
    input_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Shares_Traded': [shares_traded],
        'Turnover_Rs_Cr': [turnover]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"📌 Predicted Market Movement: **{prediction}**")
