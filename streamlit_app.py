import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="SAR–USD Exchange Rate", layout="wide")
st.title("💱 SAR–USD Exchange Rate Prediction")

# --------------------------------------------------
# Load data safely
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_pickle("SAR_USD_clean.pkl")

    # If Date is index → reset
    if "Date" not in df.columns:
        df = df.reset_index()

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Detect SAR column automatically
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(numeric_cols) != 1:
        st.error(f"Expected 1 numeric column, found: {numeric_cols}")
        st.stop()

    target_col = numeric_cols[0]

    return df, target_col

df, TARGET = load_data()

st.success(f"✅ Using exchange rate column: `{TARGET}`")

# --------------------------------------------------
# Filter stable regime
# --------------------------------------------------
df = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2022-12-31")].reset_index(drop=True)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
df["lag_1"]  = df[TARGET].shift(1)
df["lag_7"]  = df[TARGET].shift(7)
df["lag_30"] = df[TARGET].shift(30)

df["roll_7"]  = df[TARGET].rolling(7).mean()
df["roll_30"] = df[TARGET].rolling(30).mean()

df["diff_1"] = df[TARGET].diff(1)
df["diff_7"] = df[TARGET].diff(7)

df["std_7"]  = df[TARGET].rolling(7).std()
df["std_30"] = df[TARGET].rolling(30).std()

df = df.dropna().reset_index(drop=True)

features = [
    "lag_1", "lag_7", "lag_30",
    "roll_7", "roll_30",
    "diff_1", "diff_7",
    "std_7", "std_30"
]

X = df[features]
y = df[TARGET]

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
def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=20,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

model = train_model(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

st.subheader("📈 Model Performance")
c1, c2 = st.columns(2)
c1.metric("MSE", f"{mse:.10f}")
c2.metric("R²", f"{r2:.4f}")

# --------------------------------------------------
# Results Table
# --------------------------------------------------
results = pd.DataFrame({
    "Date": df["Date"].iloc[split:].values,
    "Actual SAR": y_val.values,
    "Predicted SAR": y_pred
})

st.subheader("📋 Actual vs Predicted")
st.dataframe(results.head(15))

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("📉 Prediction Chart")
st.line_chart(results.set_index("Date"))

# --------------------------------------------------
# Forecast Next Year
# --------------------------------------------------
st.subheader("🔮 1-Year Forecast")

last = df.iloc[-1:].copy()
future = []

for _ in range(365):
    X_last = last[features]
    pred = model.predict(X_last)[0]
    future.append(pred)

    last["lag_30"] = last["lag_7"]
    last["lag_7"] = last["lag_1"]
    last["lag_1"] = pred

    last["roll_7"] = pred
    last["roll_30"] = pred
    last["diff_1"] = 0
    last["diff_7"] = 0
    last["std_7"] = 0
    last["std_30"] = 0

forecast_dates = pd.date_range(
    df["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=365
)

forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Predicted SAR": future
})

st.line_chart(forecast_df.set_index("Date"))

st.success("✅ App running correctly")
