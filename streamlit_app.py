import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("exchange_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_data()
st.title("💱 Exchange Rate Viewer & Predictor")

# -------------------------------
# 2. Currency selection
# -------------------------------
currency_cols = [col for col in df.columns if col != "Date"]
currency = st.selectbox("Select a currency:", currency_cols)

# -------------------------------
# 3. Filter last 5 years
# -------------------------------
latest_year = df["Date"].dt.year.max()
first_year = latest_year - 4
hist_df = df[(df["Date"].dt.year >= first_year) & (df["Date"].dt.year <= latest_year)]

st.subheader(f"📈 {currency} Historical Rates ({first_year}-{latest_year})")
st.line_chart(hist_df.set_index("Date")[currency])

# -------------------------------
# 4. Feature engineering (lags, rolling, momentum, volatility)
# -------------------------------
data = df[["Date", currency]].copy()
data = data.sort_values("Date").reset_index(drop=True)

data["lag_1"]  = data[currency].shift(1)
data["lag_7"]  = data[currency].shift(7)
data["lag_30"] = data[currency].shift(30)
data["roll_7"]  = data[currency].rolling(7).mean()
data["roll_30"] = data[currency].rolling(30).mean()
data["diff_1"] = data[currency].diff(1)
data["diff_7"] = data[currency].diff(7)
data["std_7"]  = data[currency].rolling(7).std()
data["std_30"] = data[currency].rolling(30).std()
data = data.dropna().reset_index(drop=True)

features = ["lag_1", "lag_7", "lag_30", "roll_7", "roll_30", "diff_1", "diff_7", "std_7", "std_30"]
X = data[features]
y = data[currency]

# -------------------------------
# 5. Train/test split (time-based)
# -------------------------------
split = int(len(data)*0.8)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]

# -------------------------------
# 6. Train Random Forest
# -------------------------------
model = RandomForestRegressor(
    n_estimators=800,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------
# 7. Predict & evaluate
# -------------------------------
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

st.subheader("📊 Model Performance")
st.write(f"MSE: {mse:.6f}")
st.write(f"R²: {r2:.4f}")

# Plot actual vs predicted
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data["Date"].iloc[split:], y_val, label="Actual")
ax.plot(data["Date"].iloc[split:], y_pred, label="Predicted")
ax.set_title(f"{currency} Actual vs Predicted")
ax.set_xlabel("Date")
ax.set_ylabel(f"{currency} rate")
ax.legend()
st.pyplot(fig)

# -------------------------------
# 8. Predict next year's rate
# -------------------------------
# Use last available features
last_row = X.iloc[-1].copy()

predictions = []
for _ in range(12):  # monthly forecast for next year
    pred = model.predict([last_row])[0]
    predictions.append(pred)
    
    # Update last_row features with new prediction (simple recursive)
    last_row["lag_1"] = pred
    last_row["lag_7"] = pred
    last_row["lag_30"] = pred
    last_row["roll_7"] = pred
    last_row["roll_30"] = pred
    last_row["diff_1"] = 0
    last_row["diff_7"] = 0
    last_row["std_7"] = 0
    last_row["std_30"] = 0

st.subheader("🔮 Predicted Next Year")
st.write(f"Average predicted {currency} rate: {np.mean(predictions):.4f}")
st.line_chart(predictions)
