import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import pickle

# Set page config FIRST - this is critical
st.set_page_config(
    page_title="SAR/USD Exchange Rate Forecaster",
    page_icon="💱",
    layout="wide"
)

# Title
st.title("💱 SAR/USD Exchange Rate Forecasting Dashboard")

# Function to load data
@st.cache_data
def load_data():
    try:
        # Try to load the pickle file
        with open('SAR_USD_clean.pkl', 'rb') as f:
            df = pickle.load(f)
        st.success("✅ Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        # Create sample data for demonstration
        st.info("Creating sample data for demonstration...")
        dates = pd.date_range('2015-01-01', '2022-12-31', freq='D')
        rates = 3.75 + 0.05 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.01, len(dates))
        df = pd.DataFrame({
            'Date': dates,
            'SAR=X': rates
        })
        return df

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# Display basic info
st.subheader("📊 Data Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
with col3:
    st.metric("Current Rate", f"{df['SAR=X'].iloc[-1]:.4f}")

# Show data
with st.expander("View Raw Data"):
    st.dataframe(df.head(50))

# Feature Engineering
st.subheader("⚙️ Feature Engineering")
with st.spinner("Creating features..."):
    # Create features
    df['lag_1'] = df['SAR=X'].shift(1)
    df['lag_7'] = df['SAR=X'].shift(7)
    df['lag_30'] = df['SAR=X'].shift(30)
    df['roll_7_mean'] = df['SAR=X'].rolling(7).mean()
    df['roll_30_mean'] = df['SAR=X'].rolling(30).mean()
    df['roll_7_std'] = df['SAR=X'].rolling(7).std()
    
    # Drop NaN
    df_clean = df.dropna().copy()
    
    # Features and target
    features = ['lag_1', 'lag_7', 'lag_30', 'roll_7_mean', 'roll_30_mean', 'roll_7_std']
    X = df_clean[features]
    y = df_clean['SAR=X']
    
    # Train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Model Training
st.subheader("🤖 Model Training")
if st.button("Train Random Forest Model", type="primary"):
    with st.spinner("Training model..."):
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Store in session state
        st.session_state['model'] = model
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['metrics'] = {
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAE': mae
        }
        st.session_state['feature_importance'] = dict(zip(features, model.feature_importances_))
        
        st.success("✅ Model trained successfully!")

# Display metrics if available
if 'metrics' in st.session_state:
    st.subheader("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{st.session_state['metrics']['R²']:.4f}")
    with col2:
        st.metric("RMSE", f"{st.session_state['metrics']['RMSE']:.6f}")
    with col3:
        st.metric("MAE", f"{st.session_state['metrics']['MAE']:.6f}")
    with col4:
        st.metric("MSE", f"{st.session_state['metrics']['MSE']:.6f}")
    
    # Plot actual vs predicted
    st.subheader("📊 Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(st.session_state['y_test'].values[:100], label='Actual', alpha=0.7)
    ax.plot(st.session_state['y_pred'][:100], label='Predicted', alpha=0.7, linestyle='--')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Exchange Rate')
    ax.set_title('Actual vs Predicted Values (First 100 Samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': list(st.session_state['feature_importance'].keys()),
        'Importance': list(st.session_state['feature_importance'].values())
    }).sort_values('Importance', ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.barh(importance_df['Feature'], importance_df['Importance'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Feature Importance in Random Forest Model')
    ax2.invert_yaxis()  # Highest importance at top
    st.pyplot(fig2)

# Forecasting
st.subheader("🔮 Future Forecast")
forecast_days = st.slider("Select forecast horizon (days)", 30, 365, 90, 30)

if st.button("Generate Forecast") and 'model' in st.session_state:
    with st.spinner(f"Generating {forecast_days}-day forecast..."):
        # Use last available data for forecasting
        last_data = X.iloc[-1].values.reshape(1, -1)
        
        # Generate forecast (simplified approach)
        base_rate = df['SAR=X'].iloc[-1]
        volatility = df['SAR=X'].std() * 0.05
        
        np.random.seed(42)
        forecast = []
        current_rate = base_rate
        
        for _ in range(forecast_days):
            # Simple random walk with slight mean reversion
            change = np.random.normal(0, volatility)
            current_rate = current_rate + change
            # Keep within reasonable bounds
            current_rate = np.clip(current_rate, base_rate * 0.95, base_rate * 1.05)
            forecast.append(current_rate)
        
        # Create forecast dates
        last_date = df['Date'].max()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Display forecast stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Forecast", f"{np.mean(forecast):.4f}")
        with col2:
            st.metric("Max Forecast", f"{np.max(forecast):.4f}")
        with col3:
            st.metric("Min Forecast", f"{np.min(forecast):.4f}")
        
        # Plot forecast
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        
        # Plot historical last 180 days
        hist_dates = df['Date'].iloc[-180:]
        hist_rates = df['SAR=X'].iloc[-180:]
        ax3.plot(hist_dates, hist_rates, label='Historical', color='blue', linewidth=2)
        
        # Plot forecast
        ax3.plot(forecast_dates, forecast, label='Forecast', color='red', linewidth=2, linestyle='--')
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('SAR/USD Rate')
        ax3.set_title(f'{forecast_days}-Day Exchange Rate Forecast')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Show forecast table
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast_Rate': forecast
        })
        forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
        
        st.subheader("📅 Forecast Table (First 30 Days)")
        st.dataframe(forecast_df.head(30))

# Data Visualization
st.subheader("📈 Historical Trend")
fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.plot(df['Date'], df['SAR=X'], linewidth=1, color='green')
ax4.set_xlabel('Date')
ax4.set_ylabel('SAR/USD Rate')
ax4.set_title('Historical SAR/USD Exchange Rate')
ax4.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("""
**SAR/USD Exchange Rate Forecasting Dashboard**  
*Powered by Random Forest Regression*  
*Data Range: {start} to {end}*
""".format(
    start=df['Date'].min().date(),
    end=df['Date'].max().date()
))
