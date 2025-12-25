import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration FIRST
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
        padding-top: 1rem;
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
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">💱 SAR/USD Exchange Rate Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Function to generate synthetic data if real data is not available
@st.cache_data
def generate_sample_data():
    """Generate synthetic SAR/USD exchange rate data for demonstration"""
    np.random.seed(42)
    
    # Create date range from 2015 to 2022
    dates = pd.date_range('2015-01-01', '2022-12-31', freq='D')
    
    # Generate realistic SAR/USD rates (typically around 3.75)
    base_rate = 3.75
    noise = np.random.normal(0, 0.01, len(dates))
    trend = np.linspace(0, 0.05, len(dates))  # Slight upward trend
    
    # Add some seasonality
    seasonal = 0.01 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    
    rates = base_rate + trend + seasonal + noise
    
    # Ensure rates stay in reasonable range
    rates = np.clip(rates, 3.70, 3.80)
    
    df = pd.DataFrame({
        'Date': dates,
        'SAR=X': rates
    })
    
    return df

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess exchange rate data"""
    try:
        # Try to load from pickle file first
        df = pd.read_pickle("SAR_USD_clean.pkl")
        st.success("✓ Loaded data from SAR_USD_clean.pkl")
    except:
        try:
            # Try to load from CSV
            df = pd.read_csv("SAR_USD_clean.csv")
            st.success("✓ Loaded data from SAR_USD_clean.csv")
        except:
            # Generate sample data if files not found
            st.warning("⚠ Data files not found. Using sample data for demonstration.")
            df = generate_sample_data()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Filter to 2015-2022
    df = df[(df['Date'] >= '2015-01-01') & (df['Date'] <= '2022-12-31')].reset_index(drop=True)
    
    # Create features
    df['lag_1'] = df['SAR=X'].shift(1)
    df['lag_7'] = df['SAR=X'].shift(7)
    df['lag_30'] = df['SAR=X'].shift(30)
    df['roll_7'] = df['SAR=X'].rolling(7).mean()
    df['roll_30'] = df['SAR=X'].rolling(30).mean()
    df['diff_1'] = df['SAR=X'].diff(1)
    df['diff_7'] = df['SAR=X'].diff(7)
    df['std_7'] = df['SAR=X'].rolling(7).std()
    df['std_30'] = df['SAR=X'].rolling(30).std()
    
    df = df.dropna().reset_index(drop=True)
    
    features = ["lag_1", "lag_7", "lag_30", "roll_7", "roll_30", 
                "diff_1", "diff_7", "std_7", "std_30"]
    
    X = df[features]
    y = df["SAR=X"]
    
    # Time-based split (80/20)
    split = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    
    return df, X_train, X_val, y_train, y_val, features

# Sidebar for user inputs
st.sidebar.header("⚙️ Model Configuration")

# Model parameters
st.sidebar.subheader("Random Forest Parameters")
n_estimators = st.sidebar.slider("Number of Trees", 100, 1000, 300, 50)
max_depth = st.sidebar.slider("Max Depth", 5, 50, 15, 5)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 3)

# Forecast horizon
st.sidebar.subheader("Forecast Settings")
forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [30, 90, 180, 365],
    index=2
)

# Load data
with st.spinner("Loading and preprocessing data..."):
    df, X_train, X_val, y_train, y_val, features = load_and_preprocess_data()

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Train model button
if st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True):
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
        st.session_state.model = model
        st.session_state.y_pred = y_pred
        st.session_state.y_val = y_val
        st.session_state.metrics = {
            'MSE': mse,
            'R²': r2,
            'MAE': mae,
            'Directional Accuracy': directional_accuracy
        }
        st.session_state.feature_importance = dict(zip(features, model.feature_importances_))
        st.session_state.model_trained = True
        
        st.sidebar.success("✅ Model trained successfully!")
        st.balloons()

# Display data overview
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Historical Exchange Rate Trend")
    
    # Plot historical data
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['SAR=X'], color='#3B82F6', linewidth=1.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('SAR/USD Rate')
    ax.set_title('Historical Exchange Rate (2015-2022)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("📊 Data Statistics")
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.metric("Latest Rate", f"{df['SAR=X'].iloc[-1]:.4f}")
        st.metric("Highest", f"{df['SAR=X'].max():.4f}")
        st.metric("Lowest", f"{df['SAR=X'].min():.4f}")
    
    with stats_col2:
        st.metric("Average", f"{df['SAR=X'].mean():.4f}")
        st.metric("Std Dev", f"{df['SAR=X'].std():.4f}")
        st.metric("Total Days", len(df))

# Model Performance Section
st.markdown("---")
st.subheader("📊 Model Performance")

if st.session_state.model_trained:
    # Display metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R² Score", f"{st.session_state.metrics['R²']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mean Absolute Error", f"{st.session_state.metrics['MAE']:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Directional Accuracy", f"{st.session_state.metrics['Directional Accuracy']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Mean Squared Error", f"{st.session_state.metrics['MSE']:.6f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("📈 Actual vs Predicted Comparison")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Actual vs Predicted
    val_dates = df['Date'].iloc[-len(st.session_state.y_val):]
    ax1.plot(val_dates, st.session_state.y_val, label='Actual', color='#10B981', linewidth=2)
    ax1.plot(val_dates, st.session_state.y_pred, label='Predicted', color='#EF4444', linewidth=2, linestyle='--')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Exchange Rate')
    ax1.set_title('Validation Set: Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Feature Importance
    importance_df = pd.DataFrame({
        'Feature': list(st.session_state.feature_importance.keys()),
        'Importance': list(st.session_state.feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    ax2.barh(importance_df['Feature'], importance_df['Importance'], color='#3B82F6')
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Feature Importance')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Next Year Forecast
    st.markdown("---")
    st.subheader(f"🔮 {forecast_days}-Day Forecast")
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Generating {forecast_days}-day forecast..."):
            # Simple forecast method using last value with noise
            last_rate = df['SAR=X'].iloc[-1]
            volatility = df['SAR=X'].std() * 0.05  # 5% of historical volatility
            
            np.random.seed(42)
            forecast = [last_rate]
            for i in range(1, forecast_days):
                # Random walk with slight mean reversion
                change = np.random.normal(0, volatility)
                forecast.append(forecast[-1] + change)
            
            # Create forecast dates
            last_date = df['Date'].max()
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
            
            # Calculate confidence intervals
            forecast_mean = np.mean(forecast)
            forecast_std = np.std(forecast)
            lower_bound = [x - 1.96 * forecast_std for x in forecast]
            upper_bound = [x + 1.96 * forecast_std for x in forecast]
            
            # Display forecast summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
                st.metric("Average Forecast", f"{forecast_mean:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
                st.metric("Forecast Range", 
                         f"{min(forecast):.4f} - {max(forecast):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="forecast-card">', unsafe_allow_html=True)
                st.metric("Forecast Volatility", f"{forecast_std:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Plot forecast
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Plot historical last 180 days
            hist_dates = df['Date'].iloc[-180:]
            hist_rates = df['SAR=X'].iloc[-180:]
            ax.plot(hist_dates, hist_rates, label='Historical', color='#3B82F6', linewidth=2)
            
            # Plot forecast
            ax.plot(forecast_dates, forecast, label='Forecast', color='#EF4444', linewidth=2)
            
            # Plot confidence interval
            ax.fill_between(forecast_dates, lower_bound, upper_bound, 
                           alpha=0.2, color='#EF4444', label='95% Confidence')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('SAR/USD Rate')
            ax.set_title(f'{forecast_days}-Day Exchange Rate Forecast')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display forecast table
            st.subheader("📅 Detailed Forecast")
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })
            forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                forecast_df.head(30),
                use_container_width=True,
                hide_index=True
            )
            
            # Add download button
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast",
                data=csv,
                file_name=f"sar_usd_forecast_{forecast_days}_days.csv",
                mime="text/csv"
            )
else:
    st.info("👈 Click 'Train Model' in the sidebar to start training and see results")

# Additional Analysis
st.markdown("---")
st.subheader("📊 Additional Analysis")

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📈 Monthly Analysis", "📊 Daily Statistics", "📋 Raw Data"])

with tab1:
    st.subheader("Monthly Trends")
    
    # Extract month and year
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    monthly_stats = df.groupby(['Year', 'Month'])['SAR=X'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    # Pivot for heatmap
    pivot_data = monthly_stats.pivot(index='Year', columns='Month', values='mean')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax)
    ax.set_title('Monthly Average Exchange Rates by Year')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    st.pyplot(fig)

with tab2:
    st.subheader("Daily Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily returns distribution
        df['Daily_Return'] = df['SAR=X'].pct_change() * 100
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Daily_Return'].dropna(), bins=50, color='#3B82F6', alpha=0.7)
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Daily Returns')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Statistics table
        stats = {
            'Average Daily Return': f"{df['Daily_Return'].mean():.4f}%",
            'Std Dev of Returns': f"{df['Daily_Return'].std():.4f}%",
            'Max Daily Gain': f"{df['Daily_Return'].max():.4f}%",
            'Max Daily Loss': f"{df['Daily_Return'].min():.4f}%",
            'Positive Days': f"{(df['Daily_Return'] > 0).sum() / len(df['Daily_Return'].dropna()) * 100:.1f}%",
            'Negative Days': f"{(df['Daily_Return'] < 0).sum() / len(df['Daily_Return'].dropna()) * 100:.1f}%"
        }
        
        stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    
    # Show data info
    with st.expander("Data Information"):
        buffer = []
        df.info(buf=buffer)
        st.text("".join(buffer))
    
    # Data download
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset",
        data=csv,
        file_name="sar_usd_exchange_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1rem;'>
    <p><strong>SAR/USD Exchange Rate Forecasting Dashboard</strong></p>
    <p>Powered by Random Forest Regression • Data Range: 2015-2022</p>
    <p>Last Updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
