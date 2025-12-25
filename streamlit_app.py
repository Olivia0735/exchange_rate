import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="SAR–USD Exchange Rate Model", layout="wide")

st.title("💱 SAR–USD Exchange Rate Prediction (ML Model)")

# --------------------------------------------------
# Load data safely
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_pickle("SAR_USD_clean.pkl")

    # If Date is index, convert it
    if "Date" not in df.columns:
        df = df.reset_index()

    # Standardize column names
    df.rename(columns={"index": "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Use stable regime
    df = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2022-12-31")]

    return df.reset_index(drop=True)

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
df["lag_1"]  = df["SAR=X"].shift(1)
df["lag_7"]  = df["SAR=X"].shift(7)
df["lag_30"] = df["SAR=X"].shift(30)

df["roll_7"]  = df["SAR=X"].rolling(7).mean()
df["roll_30"] = df["SAR=X"].rolling(30).mean()

df["diff_1"] = df["SAR=X"].diff(1)
df["diff_7"] = df["SAR=X"].diff(7)

df["std_7"]  = df["SAR=X"].rolling(7).std()
df["std_30"] = df["SAR=X"].rolling(30).std()

df = df.dropna().reset_index(drop=True)

features = [
    "lag_1", "lag_7", "lag_30",
    "roll_7", "roll_30",
    "diff_1", "diff_7",
    "std_7", "std_30"
]

X = df[features]
y = df["SAR=X"]

# --------------------------------------------------
# Train / Validation split
# --------------------------------------------------
split = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]

# --------------------------------------------------
# Train Model
# --------------------------------------------------
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("MSE", f"{mse:.8f}")
col2.metric("R² Score", f"{r2:.4f}")

# --------------------------------------------------
# Actual vs Predicted Table
# --------------------------------------------------
results = pd.DataFrame({
    "Date": df["Date"].iloc[split:].values,
    "Actual SAR": y_val.values,
    "Predicted SAR": y_pred
})

st.subheader("📋 Actual vs Predicted")
st.dataframe(results.head(20))

# --------------------------------------------------
# Line Chart
# --------------------------------------------------
st.subheader("📉 Prediction Visualization")
st.line_chart(results.set_index("Date")[["Actual SAR", "Predicted SAR"]])

# --------------------------------------------------
# Next 1-Year Forecast (Iterative)
# --------------------------------------------------
st.subheader("🔮 1-Year Forecast")

last_row = df.iloc[-1:].copy()
future_predictions = []

for i in range(365):
    X_last = last_row[features]
    next_value = model.predict(X_last)[0]

    future_predictions.append(next_value)

    # Shift features
    last_row["lag_30"] = last_row["lag_7"]
    last_row["lag_7"] = last_row["lag_1"]
    last_row["lag_1"] = next_value

    last_row["roll_7"] = next_value
    last_row["roll_30"] = next_value
    last_row["diff_1"] = 0
    last_row["diff_7"] = 0
    last_row["std_7"] = 0
    last_row["std_30"] = 0

future_dates = pd.date_range(
    start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=365
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted SAR": future_predictions
})

st.line_chart(forecast_df.set_index("Date"))

st.success("✅ App loaded successfully")
