import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# App title
# --------------------------------------------------
st.title("SAR–USD Exchange Rate Prediction")
st.markdown("Random Forest model using lag, momentum, and volatility features")

# --------------------------------------------------
# Load cleaned SAR data
# --------------------------------------------------
df = pd.read_csv("SAR_USD_clean.pkl")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Filter recent regime
df = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2022-12-31")].reset_index(drop=True)

# --------------------------------------------------
# Feature engineering
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
# Train / validation split
# --------------------------------------------------
split = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y.iloc[:split], y.iloc[split:]

# --------------------------------------------------
# Train model
# --------------------------------------------------
model = RandomForestRegressor(
    n_estimators=800,
    max_depth=20,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
y_pred = model.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

st.subheader("Model Accuracy")
st.metric("Mean Squared Error (MSE)", f"{mse:.8f}")
st.metric("R² Score", f"{r2:.3f}")

# --------------------------------------------------
# Actual vs Predicted Plot
# --------------------------------------------------
st.subheader("Actual vs Predicted SAR")

fig, ax = plt.subplots()
ax.plot(y_val.values[:200], label="Actual")
ax.plot(y_pred[:200], label="Predicted")
ax.set_ylabel("SAR per USD")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------
# Feature Importance
# --------------------------------------------------
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

# --------------------------------------------------
# Forecast next year (252 trading days)
# --------------------------------------------------
st.subheader("Forecast: Next Year (2023)")

last_row = df.iloc[-1][features].values.reshape(1, -1)

future_predictions = []
current_features = last_row.copy()

for _ in range(252):
    next_pred = model.predict(current_features)[0]
    future_predictions.append(next_pred)

    # Shift lag features
    current_features[0, 2] = current_features[0, 1]
    current_features[0, 1] = current_features[0, 0]
    current_features[0, 0] = next_pred

# Display forecast
st.write("Average predicted SAR for 2023:")
st.success(f"{np.mean(future_predictions):.4f} SAR per USD")

# --------------------------------------------------
# Show sample data
# --------------------------------------------------
st.subheader("Sample Predictions")
preview = pd.DataFrame({
    "Actual SAR": y_val.values[:10],
    "Predicted SAR": y_pred[:10]
})
st.dataframe(preview)
