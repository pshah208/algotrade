import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
import requests
import time
warnings.filterwarnings('ignore')

# Configure requests to handle network issues
requests.packages.urllib3.disable_warnings()

# Alternative: Use sample data if connection fails
def get_sample_data(ticker):
    """Generate sample data when network fails"""
    st.warning("Using sample data due to network issues")
    
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends
    
    # Generate realistic stock price data
    np.random.seed(42)
    initial_price = 150.0
    price_changes = np.random.normal(0, 2, len(dates))
    prices = [initial_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] + change
        prices.append(max(new_price, 1))  # Ensure price doesn't go negative
    
    df = pd.DataFrame({
        'Date': dates[:len(prices)],
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(prices))
    })
    
    st.info(f"Generated {len(df)} sample data points for {ticker}")
    return df# Alternative: Use sample data if connection fails
def get_sample_data(ticker):
    """Generate sample data when network fails"""
    st.warning("Using sample data due to network issues")
    
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends
    
    # Generate realistic stock price data
    np.random.seed(42)
    initial_price = 150.0
    price_changes = np.random.normal(0, 2, len(dates))
    prices = [initial_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] + change
        prices.append(max(new_price, 1))  # Ensure price doesn't go negative
    
    df = pd.DataFrame({
        'Date': dates[:len(prices)],
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(prices))
    })
    
    st.info(f"Generated {len(df)} sample data points for {ticker}")
    return df

st.title("Algorithmic Trading Price Predictor - Fixed Version")

# Initialize session state for ticker
if 'ticker_input' not in st.session_state:
    st.session_state.ticker_input = "AAPL"

ticker = st.text_input("Enter ticker symbol", value=st.session_state.ticker_input).upper()

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="1y", interval="1d"):
    """Fetch stock data with multiple retry strategies - let yfinance handle sessions"""
    import time
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            st.info(f"Fetching data for {ticker} (Attempt {attempt + 1}/{max_retries})...")
            
            # Try different approaches based on attempt - no custom sessions
            if attempt == 0:
                # First attempt: standard download (let yfinance handle session)
                df = yf.download(
                    ticker, 
                    period=period, 
                    interval=interval, 
                    progress=False, 
                    auto_adjust=False,
                    threads=False  # Disable threading which can cause issues
                )
            elif attempt == 1:
                # Second attempt: use ticker object method (no custom session)
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(
                    period=period,
                    interval=interval,
                    auto_adjust=False
                )
            else:
                # Third attempt: try shorter period to reduce load
                shorter_period = "6mo" if period == "1y" else "3mo"
                st.warning(f"Trying shorter period: {shorter_period}")
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(
                    period=shorter_period,
                    interval=interval,
                    auto_adjust=False
                )
            
            if df.empty:
                if attempt < max_retries - 1:
                    st.warning(f"No data received for {ticker}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    st.error(f"No data found for ticker {ticker} after {max_retries} attempts")
                    return pd.DataFrame()
            
            # Fix column names - yfinance returns MultiIndex columns sometimes
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Ensure we have the required columns
            if 'Close' not in df.columns:
                st.error(f"Missing 'Close' column. Available columns: {df.columns.tolist()}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            st.success(f"Successfully fetched {len(df)} rows for {ticker}")
            st.write(f"Available columns: {df.columns.tolist()}")
            st.write(f"Data shape: {df.shape}")
            st.write(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            st.warning(f"Error on attempt {attempt + 1}: {error_msg}")
            
            if attempt < max_retries - 1:
                st.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                st.error(f"Final attempt failed: {error_msg}")
    
    st.error(f"Failed to fetch data for {ticker} after {max_retries} attempts")
    
    # Offer to use sample data
    if st.button(f"Use Sample Data for {ticker}", key="sample_data_btn"):
        return get_sample_data(ticker)
    
    st.info("Click the button above to use sample data, or try a different ticker symbol.")
    return pd.DataFrame()

def add_technical_indicators(df):
    """Add technical indicators with proper error handling"""
    df = df.copy()
    st.write("Columns at start of add_technical_indicators:", df.columns.tolist())

    if 'Close' not in df.columns:
        st.error("ERROR: 'Close' column missing from data.")
        return df

    try:
        # Simple Moving Average
        df['SMA_14'] = df['Close'].rolling(window=14, min_periods=1).mean()
        
        # Exponential Moving Average
        df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20, min_periods=1).mean()
        std_20 = df['Close'].rolling(window=20, min_periods=1).std()
        df['Bollinger_Upper'] = sma_20 + (std_20 * 2)
        df['Bollinger_Lower'] = sma_20 - (std_20 * 2)

        # Fill any remaining NaN values with forward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        st.write("Columns after adding indicators:", df.columns.tolist())
        
    except Exception as e:
        st.error(f"Error adding technical indicators: {str(e)}")
        
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    try:
        df = add_technical_indicators(df)
        df = df.copy()
        
        # Add lagged features
        df['close_lag1'] = df['Close'].shift(1)
        df['close_lag2'] = df['Close'].shift(2)
        
        # Add volume if available
        if 'Volume' in df.columns:
            df['volume_sma_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()

        st.write("Columns after adding lags:", df.columns.tolist())

        # Define feature columns
        features = ['close_lag1', 'close_lag2', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
        
        # Add volume feature if available
        if 'volume_sma_5' in df.columns:
            features.append('volume_sma_5')

        # Check for missing columns
        missing_cols = [col for col in features + ['Close'] if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns before processing: {missing_cols}")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Drop rows with NaN values
        df_clean = df.dropna(subset=features + ['Close'])
        st.write(f"Data shape after dropna: {df_clean.shape}")
        
        if len(df_clean) < 50:  # Need minimum data for training
            st.error(f"Insufficient data after cleaning: {len(df_clean)} rows. Need at least 50 rows.")
            return pd.DataFrame(), pd.Series(dtype=float)

        X = df_clean[features]
        y = df_clean['Close']
        
        # Check for infinite values
        if np.isinf(X.values).any() or np.isinf(y.values).any():
            st.error("Infinite values detected in features or target")
            return pd.DataFrame(), pd.Series(dtype=float)
            
        return X, y
        
    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return pd.DataFrame(), pd.Series(dtype=float)

@st.cache_resource
def train_model(X, y):
    """Train the machine learning model with comprehensive evaluation"""
    try:
        # Use temporal split (no shuffling for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        st.write(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            max_depth=10,  # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        # Calculate comprehensive metrics for training set
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        train_r2 = r2_score(y_train, train_preds)
        
        # Calculate comprehensive metrics for test set
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        # Calculate percentage errors for better interpretation
        train_mape = np.mean(np.abs((y_train - train_preds) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
        
        # Store all metrics in a dictionary
        metrics = {
            'train': {
                'rmse': train_rmse,
                'mae': train_mae,
                'r2': train_r2,
                'mape': train_mape
            },
            'test': {
                'rmse': test_rmse,
                'mae': test_mae,
                'r2': test_r2,
                'mape': test_mape,
                'actual': y_test.values,
                'predicted': test_preds,
                'dates': X_test.index
            }
        }
        
        # Display metrics in a nice format
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Set Performance:**")
            st.metric("RMSE", f"${train_rmse:.2f}")
            st.metric("MAE", f"${train_mae:.2f}")
            st.metric("R² Score", f"{train_r2:.4f}")
            st.metric("MAPE", f"{train_mape:.2f}%")
        
        with col2:
            st.write("**Test Set Performance:**")
            st.metric("RMSE", f"${test_rmse:.2f}")
            st.metric("MAE", f"${test_mae:.2f}")
            st.metric("R² Score", f"{test_r2:.4f}")
            st.metric("MAPE", f"{test_mape:.2f}%")
        
        # Performance interpretation
        if test_r2 > 0.8:
            performance_msg = "🟢 Excellent model performance!"
        elif test_r2 > 0.6:
            performance_msg = "🟡 Good model performance"
        elif test_r2 > 0.4:
            performance_msg = "🟠 Moderate model performance"
        else:
            performance_msg = "🔴 Poor model performance - consider feature engineering"
        
        st.info(f"**Model Assessment:** {performance_msg}")
        
        # Check for overfitting
        r2_diff = train_r2 - test_r2
        if r2_diff > 0.1:
            st.warning(f"⚠️ Potential overfitting detected (Training R²: {train_r2:.3f}, Test R²: {test_r2:.3f})")
        elif r2_diff < 0.05:
            st.success("✅ Model shows good generalization (low overfitting)")
        
        return model, metrics
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None

def predict_next(model, last_features):
    """Predict next price"""
    try:
        pred = model.predict(np.array(last_features).reshape(1, -1))
        return pred[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Main execution
if st.button("Fetch, Train & Predict"):
    with st.spinner("Fetching data..."):
        data = fetch_data(ticker)
        
    if data.empty:
        st.error(f"No data found for ticker {ticker}.")
    else:
        st.success(f"Fetched {len(data)} rows for {ticker}")
        
        with st.spinner("Preparing features..."):
            X, y = prepare_features(data)
            
        if len(X) == 0:
            st.error("Not enough data or missing columns for features.")
        else:
            with st.spinner("Training model..."):
                result = train_model(X, y)
                
            if result[0] is not None:  # Check if model training was successful
                model, metrics = result
                test_rmse = metrics['test']['rmse']
                st.success(f"Model trained successfully! Test RMSE: ${test_rmse:.2f}")

                # Create model evaluation visualizations
                st.subheader("📈 Model Evaluation Plots")
                
                # Create a 2x2 subplot layout
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Plot 1: Actual vs Predicted (Test Set)
                actual = metrics['test']['actual']
                predicted = metrics['test']['predicted']
                
                ax1.scatter(actual, predicted, alpha=0.6, color='blue', s=30)
                
                # Add perfect prediction line (y=x)
                min_val = min(actual.min(), predicted.min())
                max_val = max(actual.max(), predicted.max())
                ax1.plot([min_val, max_val], [min_val, max_val], 'r--', 
                        linewidth=2, label='Perfect Prediction (y=x)')
                
                # Add trend line
                z = np.polyfit(actual, predicted, 1)
                p = np.poly1d(z)
                ax1.plot(actual, p(actual), 'g--', alpha=0.8, 
                        label=f'Trend Line (slope={z[0]:.3f})')
                
                ax1.set_xlabel('Actual Prices ($)')
                ax1.set_ylabel('Predicted Prices ($)')
                ax1.set_title(f'Actual vs Predicted Prices\n(R² = {metrics["test"]["r2"]:.3f})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Add text box with metrics
                textstr = f'RMSE: ${metrics["test"]["rmse"]:.2f}\nMAE: ${metrics["test"]["mae"]:.2f}\nMAPE: {metrics["test"]["mape"]:.2f}%'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
                
                # Plot 2: Residuals Plot (Test Set)
                residuals = actual - predicted
                ax2.scatter(predicted, residuals, alpha=0.6, color='red', s=30)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax2.set_xlabel('Predicted Prices ($)')
                ax2.set_ylabel('Residuals ($)')
                ax2.set_title('Residuals Plot (Test Set)')
                ax2.grid(True, alpha=0.3)
                
                # Add residual statistics
                residual_std = np.std(residuals)
                ax2.axhline(y=residual_std, color='orange', linestyle='--', alpha=0.7, label=f'+1σ ({residual_std:.2f})')
                ax2.axhline(y=-residual_std, color='orange', linestyle='--', alpha=0.7, label=f'-1σ ({-residual_std:.2f})')
                ax2.legend()
                
                # Plot 3: Time Series of Predictions vs Actual (if we have enough test data)
                if len(actual) > 10:
                    test_indices = range(len(actual))
                    ax3.plot(test_indices, actual, label='Actual', color='blue', linewidth=2)
                    ax3.plot(test_indices, predicted, label='Predicted', color='red', linewidth=2, alpha=0.7)
                    ax3.fill_between(test_indices, actual, predicted, alpha=0.2, color='gray')
                    ax3.set_xlabel('Time Steps (Test Period)')
                    ax3.set_ylabel('Price ($)')
                    ax3.set_title('Time Series: Actual vs Predicted (Test Set)')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                else:
                    ax3.text(0.5, 0.5, 'Insufficient test data\nfor time series plot', 
                            ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                    ax3.set_title('Time Series Plot (Insufficient Data)')
                
                # Plot 4: Error Distribution
                ax4.hist(residuals, bins=min(20, len(residuals)//2), alpha=0.7, color='skyblue', edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
                ax4.axvline(x=np.mean(residuals), color='green', linestyle='-', linewidth=2, 
                           label=f'Mean Error ({np.mean(residuals):.2f})')
                ax4.set_xlabel('Prediction Error ($)')
                ax4.set_ylabel('Frequency')
                ax4.set_title(f'Error Distribution\n(Std: ${np.std(residuals):.2f})')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Additional insights
                st.subheader("🔍 Model Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Prediction accuracy analysis
                    within_5_percent = np.sum(np.abs(residuals) <= actual * 0.05) / len(actual) * 100
                    within_10_percent = np.sum(np.abs(residuals) <= actual * 0.10) / len(actual) * 100
                    
                    st.metric("Predictions within 5%", f"{within_5_percent:.1f}%")
                    st.metric("Predictions within 10%", f"{within_10_percent:.1f}%")
                
                with col2:
                    # Directional accuracy (for next-day predictions)
                    if len(actual) > 1:
                        actual_direction = np.diff(actual) > 0
                        predicted_direction = np.diff(predicted) > 0
                        directional_accuracy = np.sum(actual_direction == predicted_direction) / len(actual_direction) * 100
                        st.metric("Directional Accuracy", f"{directional_accuracy:.1f}%")
                    
                    largest_error = np.abs(residuals).max()
                    st.metric("Largest Error", f"${largest_error:.2f}")
                
                with col3:
                    # Model consistency
                    error_consistency = 1 - (np.std(residuals) / np.mean(np.abs(residuals)))
                    st.metric("Error Consistency", f"{error_consistency:.3f}")
                    
                    # Bias detection
                    mean_error = np.mean(residuals)
                    bias_status = "📈 Overestimating" if mean_error > 0 else "📉 Underestimating" if mean_error < 0 else "✅ Unbiased"
                    st.metric("Bias Status", bias_status)

                # Make prediction
                last_features = X.iloc[-1].values
                predicted_price = predict_next(model, last_features)
                
                if predicted_price is not None:
                    current_price = data['Close'].iloc[-1]
                    price_change = predicted_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    st.success(f"**Predicted next closing price for {ticker}: ${predicted_price:.2f}**")
                    st.info(f"Current price: ${current_price:.2f}")
                    st.info(f"Predicted change: ${price_change:.2f} ({price_change_pct:+.2f}%)")

                    # Create visualization
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Plot 1: Historical prices and prediction
                    ax1.plot(data['Date'], data['Close'], label='Historical Close', linewidth=1.5)
                    
                    # Add prediction point
                    last_date = data['Date'].iloc[-1]
                    if hasattr(last_date, 'to_pydatetime'):
                        predicted_date = last_date.to_pydatetime() + timedelta(days=1)
                    else:
                        predicted_date = last_date + timedelta(days=1)
                    
                    ax1.scatter(predicted_date, predicted_price, color='red', 
                              label=f'Predicted Close (${predicted_price:.2f})', 
                              s=100, marker='*', zorder=5)
                    
                    ax1.set_title(f"{ticker} Closing Prices and Prediction")
                    ax1.set_xlabel("Date")
                    ax1.set_ylabel("Price ($)")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Recent performance focus
                    recent_data = data.tail(60)  # Last 60 days
                    ax2.plot(recent_data['Date'], recent_data['Close'], 
                            label='Recent Close', linewidth=2, color='blue')
                    ax2.scatter(predicted_date, predicted_price, color='red', 
                              label=f'Predicted', s=100, marker='*', zorder=5)
                    
                    ax2.set_title(f"Recent {ticker} Performance (Last 60 Days)")
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Price ($)")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Show feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        st.subheader("Feature Importance")
                        st.dataframe(feature_importance)

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info("""
This app predicts stock prices using machine learning.

**Features used:**
- Lagged closing prices
- Simple Moving Average (14 days)
- Exponential Moving Average (14 days)  
- RSI (14 days)
- MACD
- Bollinger Bands

**Model:** Random Forest Regressor

**Note:** This is for educational purposes only. 
Do not use for actual trading decisions.
""")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
2. Click 'Fetch, Train & Predict'
3. Wait for the model to train and make predictions
4. View the results and visualizations
""")

st.sidebar.header("Troubleshooting")
st.sidebar.write("""
**If you get timeout errors:**
- Try popular tickers: AAPL, MSFT, GOOGL, TSLA
- Check your internet connection
- Wait a few minutes and try again
- Use the sample data option if offered

**Common issues:**
- Network connectivity problems
- Yahoo Finance server overload
- Invalid ticker symbols
""")

# Add network status check
st.sidebar.header("Network Test")
if st.sidebar.button("Test Yahoo Finance Connection"):
    try:
        # Simple test using yfinance itself
        test_ticker = yf.Ticker("AAPL")
        test_info = test_ticker.info
        if test_info and 'symbol' in test_info:
            st.sidebar.success("✅ Yahoo Finance is accessible")
        else:
            st.sidebar.warning("⚠️ Yahoo Finance connection unclear")
    except Exception as e:
        st.sidebar.error(f"❌ Cannot reach Yahoo Finance: {str(e)}")

# Quick data options
st.sidebar.header("Quick Options")
popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
selected_ticker = st.sidebar.selectbox("Select Popular Ticker:", [""] + popular_tickers)
if selected_ticker:
    # Update the main ticker input when sidebar selection changes
    st.session_state.ticker_input = selected_ticker
