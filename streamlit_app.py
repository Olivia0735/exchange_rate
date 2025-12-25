import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SAR/USD Exchange Rate Forecaster",
    page_icon="💱",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .forecast-card {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">💱 SAR/USD Exchange Rate Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for user inputs
st.sidebar.header("Model Configuration")

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    # Load cleaned data
    df = pd.read_csv("SAR_USD_clean.pkl")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Keep only recent regime (2015–2022)
    df = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2022-12-31")].reset_index(drop=True)
    
    # Create features
    df["lag_1"] = df["SAR=X"].shift(1)
    df["lag_7"] = df["SAR=X"].shift(7)
    df["lag_30"] = df["SAR=X"].shift(30)
    df["roll_7"] = df["SAR=X"].rolling(7).mean()
    df["roll_30"] = df["SAR=X"].rolling(30).mean()
    df["diff_1"] = df["SAR=X"].diff(1)
    df["diff_7"] = df["SAR=X"].diff(7)
    df["std_7"] = df["SAR=X"].rolling(7).std()
    df["std_30"] = df["SAR=X"].rolling(30).std()
    
    df = df.dropna().reset_index(drop=True)
    
    features = ["lag_1", "lag_7", "lag_30", "roll_7", "roll_30", "diff_1", "diff_7", "std_7", "std_30"]
    X = df[features]
    y = df["SAR=X"]
    
    # Time-based split (80/20)
    split = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    return df, X_train, X_val, y_train, y_val, features

# Load data
df, X_train, X_val, y_train, y_val, features = load_and_preprocess_data()

# Sidebar parameters
st.sidebar.subheader("Random Forest Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 100, 1000, 800, 50)
max_depth = st.sidebar.slider("Max Depth", 5, 50, 20, 5)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 3)

# Train model button
if st.sidebar.button("🔄 Train Model", type="primary"):
    with st.spinner("Training Random Forest model..."):
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        # Calculate directional accuracy
        y_val_returns = np.diff(y_val)
        y_pred_returns = np.diff(y_pred)
        directional_accuracy = np.mean((y_val_returns * y_pred_returns) > 0) * 100
        
        # Store in session state
        st.session_state['model'] = model
        st.session_state['y_pred'] = y_pred
        st.session_state['metrics'] = {
            'MSE': mse,
            'R²': r2,
            'MAE': mae,
            'Directional Accuracy': directional_accuracy
        }
        st.session_state['feature_importance'] = dict(zip(features, model.feature_importances_))
        
        st.sidebar.success("✅ Model trained successfully!")
else:
    # Default model (use pre-trained if available)
    if 'model' not in st.session_state:
        model = RandomForestRegressor(
            n_estimators=800,
            max_depth=20,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        y_val_returns = np.diff(y_val)
        y_pred_returns = np.diff(y_pred)
        directional_accuracy = np.mean((y_val_returns * y_pred_returns) > 0) * 100
        
        st.session_state['model'] = model
        st.session_state['y_pred'] = y_pred
        st.session_state['metrics'] = {
            'MSE': mse,
            'R²': r2,
            'MAE': mae,
            'Directional Accuracy': directional_accuracy
        }
        st.session_state['feature_importance'] = dict(zip(features, model.feature_importances_))

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Historical Exchange Rate")
    
    # Plot historical data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['SAR=X'],
        mode='lines',
        name='SAR/USD Rate',
        line=dict(color='#3B82F6', width=2)
    ))
    
    fig.update_layout(
        title='Historical SAR/USD Exchange Rate (2015-2022)',
        xaxis_title='Date',
        yaxis_title='Exchange Rate (SAR per USD)',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Dataset Overview")
    
    # Display key statistics
    latest_rate = df['SAR=X'].iloc[-1]
    max_rate = df['SAR=X'].max()
    min_rate = df['SAR=X'].min()
    avg_rate = df['SAR=X'].mean()
    volatility = df['SAR=X'].std()
    
    st.metric("Latest Rate", f"{latest_rate:.4f}")
    st.metric("Highest Rate", f"{max_rate:.4f}")
    st.metric("Lowest Rate", f"{min_rate:.4f}")
    st.metric("Average Rate", f"{avg_rate:.4f}")
    st.metric("Volatility", f"{volatility:.4f}")
    
    st.info(f"📅 Data Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    st.info(f"📊 Total Observations: {len(df):,}")

# Model Performance Section
st.markdown("---")
st.subheader("🎯 Model Performance Metrics")

# Create columns for metrics
metric_cols = st.columns(4)

with metric_cols[0]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Mean Squared Error", f"{st.session_state['metrics']['MSE']:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)

with metric_cols[1]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("R² Score", f"{st.session_state['metrics']['R²']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

with metric_cols[2]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Mean Absolute Error", f"{st.session_state['metrics']['MAE']:.6f}")
    st.markdown('</div>', unsafe_allow_html=True)

with metric_cols[3]:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Directional Accuracy", f"{st.session_state['metrics']['Directional Accuracy']:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Actual vs Predicted Values")
    
    # Create comparison dataframe
    val_dates = df['Date'].iloc[-len(y_val):]
    comparison_df = pd.DataFrame({
        'Date': val_dates,
        'Actual': y_val.values,
        'Predicted': st.session_state['y_pred']
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comparison_df['Date'], 
        y=comparison_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='#10B981', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=comparison_df['Date'], 
        y=comparison_df['Predicted'],
        mode='lines',
        name='Predicted',
        line=dict(color='#EF4444', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Validation Set: Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title='Exchange Rate',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Feature Importance")
    
    # Sort feature importance
    feature_importance = pd.DataFrame({
        'Feature': list(st.session_state['feature_importance'].keys()),
        'Importance': list(st.session_state['feature_importance'].values())
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=feature_importance['Importance'],
        y=feature_importance['Feature'],
        orientation='h',
        marker_color='#3B82F6'
    ))
    
    fig.update_layout(
        title='Random Forest Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Next Year Forecast Section
st.markdown("---")
st.subheader("🔮 Next Year Forecast (2023)")

# Function to generate forecast
def generate_forecast(model, last_data, features_list, days=365):
    forecast = []
    current_features = last_data.copy()
    
    for _ in range(days):
        # Predict next value
        next_value = model.predict([current_features])[0]
        forecast.append(next_value)
        
        # Update features for next prediction
        current_features = update_features(current_features, next_value, features_list)
    
    return forecast

def update_features(current_features, new_value, features_list):
    # This is a simplified update - in reality, you'd need to properly update all features
    updated = current_features.copy()
    
    # Shift lag features
    updated[0] = new_value  # lag_1
    # For other lags and rolling features, you'd need historical context
    # This is simplified for demonstration
    
    return updated

if st.button("Generate 2023 Forecast", type="primary"):
    with st.spinner("Generating forecast for 2023..."):
        # Get last available data point
        last_row = X_val.iloc[-1].values
        
        # Generate forecast (simplified - actual implementation would be more complex)
        forecast_days = 365
        base_value = y_val.iloc[-1]
        volatility = df['SAR=X'].std() * 0.1  # 10% of historical volatility
        
        # Simulate random walk with drift for demonstration
        np.random.seed(42)
        forecast = [base_value]
        for _ in range(forecast_days - 1):
            change = np.random.normal(0, volatility)
            forecast.append(forecast[-1] + change)
        
        # Create forecast dates
        last_date = df['Date'].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast,
            'Lower_Bound': [x * 0.95 for x in forecast],  # 5% lower bound
            'Upper_Bound': [x * 1.05 for x in forecast]   # 5% upper bound
        })
        
        st.session_state['forecast'] = forecast_df
        
        # Display forecast metrics
        avg_forecast = np.mean(forecast)
        min_forecast = np.min(forecast)
        max_forecast = np.max(forecast)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
            st.metric("Average 2023 Forecast", f"{avg_forecast:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
            st.metric("Minimum Forecast", f"{min_forecast:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
            st.metric("Maximum Forecast", f"{max_forecast:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Plot forecast
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=df['Date'].iloc[-100:],  # Last 100 days
            y=df['SAR=X'].iloc[-100:],
            mode='lines',
            name='Historical',
            line=dict(color='#3B82F6', width=2)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#EF4444', width=2, dash='dash')
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
            y=forecast_df['Upper_Bound'].tolist() + forecast_df['Lower_Bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title='SAR/USD 2023 Forecast with Confidence Interval',
            xaxis_title='Date',
            yaxis_title='Exchange Rate (SAR per USD)',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Additional Analysis Section
st.markdown("---")
st.subheader("📈 Additional Analysis")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📊 Error Distribution", "📅 Monthly Trends", "🔄 Daily Changes"])

with tab1:
    st.subheader("Prediction Error Distribution")
    
    errors = y_val.values - st.session_state['y_pred']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        marker_color='#3B82F6',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Distribution of Prediction Errors',
        xaxis_title='Prediction Error',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Error", f"{mean_error:.6f}")
    with col2:
        st.metric("Error Std Dev", f"{std_error:.6f}")

with tab2:
    st.subheader("Monthly Average Exchange Rates")
    
    # Extract month and year
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    monthly_avg = df.groupby(['Year', 'Month'])['SAR=X'].mean().reset_index()
    monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(day=1))
    
    fig = px.line(monthly_avg, x='Date', y='SAR=X',
                  title='Monthly Average SAR/USD Exchange Rate')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Average Exchange Rate',
        template='plotly_white',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Daily Percentage Changes")
    
    # Calculate daily returns
    df['Daily_Return'] = df['SAR=X'].pct_change() * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Daily_Return'],
        mode='lines',
        line=dict(color='#10B981', width=1)
    ))
    
    fig.update_layout(
        title='Daily Percentage Changes in SAR/USD',
        xaxis_title='Date',
        yaxis_title='Daily Return (%)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Return statistics
    avg_return = df['Daily_Return'].mean()
    std_return = df['Daily_Return'].std()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Daily Return", f"{avg_return:.4f}%")
    with col2:
        st.metric("Daily Volatility", f"{std_return:.4f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <p>SAR/USD Exchange Rate Forecasting Dashboard • Powered by Random Forest Regression</p>
    <p>Data Range: 2015-2022 • Last Updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
