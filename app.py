import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

ticker = st.text_input("Enter ticker symbol", "AAPL").upper()

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="1y", interval="1d"):
    """Fetch stock data with multiple retry strategies and timeout handling"""
    import time
    import requests
    
    # Configure session with longer timeout and retry logic
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            st.info(f"Fetching data for {ticker} (Attempt {attempt + 1}/{max_retries})...")
            
            # Create ticker object with custom session
            ticker_obj = yf.Ticker(ticker, session=session)
            
            # Try different approaches based on attempt
            if attempt == 0:
                # First attempt: standard download with longer timeout
                df = yf.download(
                    ticker, 
                    period=period, 
                    interval=interval, 
                    progress=False, 
                    auto_adjust=False,
                    timeout=30,  # Increase timeout to 30 seconds
                    threads=False  # Disable threading which can cause issues
                )
            elif attempt == 1:
                # Second attempt: use ticker object method
                df = ticker_obj.history(
                    period=period,
                    interval=interval,
                    auto_adjust=False,
                    timeout=30
                )
            else:
                # Third attempt: try shorter period to reduce load
                shorter_period = "6mo" if period == "1y" else "3mo"
                st.warning(f"Trying shorter period: {shorter_period}")
                df = ticker_obj.history(
                    period=shorter_period,
                    interval=interval,
                    auto_adjust=False,
                    timeout=30
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
            required_cols = ['Close']  # Minimum required
            available_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]
            
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
            
        except requests.exceptions.Timeout:
            st.warning(f"Timeout error on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            continue
            
        except requests.exceptions.ConnectionError:
            st.warning(f"Connection error on attempt {attempt + 1}. Retrying in {retry_delay} seconds...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            continue
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                st.warning(f"Network error on attempt {attempt + 1}: {error_msg}")
                if attempt < max_retries - 1:
                    st.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
            else:
                st.error(f"Error fetching data: {error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
    
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
    """Train the machine learning model"""
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
        
        # Make predictions and calculate RMSE
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        st.write(f"Training RMSE: {train_rmse:.2f}")
        st.write(f"Test RMSE: {test_rmse:.2f}")
        
        return model, test_rmse, X_test.index
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

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
                model, rmse, test_indices = result
                st.success(f"Model trained successfully! Test RMSE: {rmse:.2f}")

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
        response = requests.get("https://finance.yahoo.com", timeout=10)
        if response.status_code == 200:
            st.sidebar.success("✅ Yahoo Finance is accessible")
        else:
            st.sidebar.error(f"❌ Yahoo Finance returned status {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"❌ Cannot reach Yahoo Finance: {str(e)}")

# Quick data options
st.sidebar.header("Quick Options")
popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
selected_ticker = st.sidebar.selectbox("Select Popular Ticker:", [""] + popular_tickers)
if selected_ticker and st.sidebar.button("Use Selected Ticker"):
    st.rerun()
