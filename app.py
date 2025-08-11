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
    """Add comprehensive technical indicators with proper error handling"""
    df = df.copy()
    st.write("Columns at start of add_technical_indicators:", df.columns.tolist())

    if 'Close' not in df.columns:
        st.error("ERROR: 'Close' column missing from data.")
        return df

    try:
        # Basic Moving Averages (Multiple timeframes)
        df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['SMA_14'] = df['Close'].rolling(window=14, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Moving Average Crossovers (Strong predictive signals)
        df['SMA_5_14_ratio'] = df['SMA_5'] / (df['SMA_14'] + 1e-10)
        df['SMA_14_50_ratio'] = df['SMA_14'] / (df['SMA_50'] + 1e-10)
        df['EMA_14_50_ratio'] = df['EMA_14'] / (df['EMA_50'] + 1e-10)
        
        # Price Position Relative to Moving Averages
        df['price_above_SMA14'] = (df['Close'] > df['SMA_14']).astype(int)
        df['price_above_SMA50'] = (df['Close'] > df['SMA_50']).astype(int)
        
        # Volatility Features
        df['volatility_10'] = df['Close'].rolling(window=10, min_periods=1).std()
        df['volatility_30'] = df['Close'].rolling(window=30, min_periods=1).std()
        df['volatility_ratio'] = df['volatility_10'] / (df['volatility_30'] + 1e-10)
        
        # Price change features (momentum)
        df['returns_1d'] = df['Close'].pct_change(1)
        df['returns_3d'] = df['Close'].pct_change(3)
        df['returns_7d'] = df['Close'].pct_change(7)
        df['returns_14d'] = df['Close'].pct_change(14)
        
        # Cumulative returns (trend strength)
        df['cum_returns_5d'] = df['returns_1d'].rolling(5, min_periods=1).sum()
        df['cum_returns_10d'] = df['returns_1d'].rolling(10, min_periods=1).sum()
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD with signal line
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands with additional features
        for period in [20, 50]:
            sma = df['Close'].rolling(window=period, min_periods=1).mean()
            std = df['Close'].rolling(window=period, min_periods=1).std()
            df[f'BB_upper_{period}'] = sma + (std * 2)
            df[f'BB_lower_{period}'] = sma - (std * 2)
            df[f'BB_width_{period}'] = (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']) / sma
            df[f'BB_position_{period}'] = (df['Close'] - df[f'BB_lower_{period}']) / (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])
        
        # Volume-based features (if volume available)
        if 'Volume' in df.columns:
            df['volume_sma_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
            df['volume_sma_30'] = df['Volume'].rolling(window=30, min_periods=1).mean()
            df['volume_ratio'] = df['Volume'] / (df['volume_sma_10'] + 1e-10)
            df['price_volume'] = df['Close'] * df['Volume']
            df['volume_price_trend'] = df['price_volume'].rolling(window=5, min_periods=1).mean()
            
            # On-Balance Volume (OBV)
            df['OBV'] = (df['Volume'] * np.sign(df['Close'].diff())).cumsum()
            df['OBV_sma'] = df['OBV'].rolling(window=10, min_periods=1).mean()
        
        # High-Low features (if available)
        if 'High' in df.columns and 'Low' in df.columns:
            df['daily_range'] = df['High'] - df['Low']
            df['daily_range_pct'] = df['daily_range'] / df['Close']
            df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-10)
            
            # True Range and Average True Range (ATR)
            df['prev_close'] = df['Close'].shift(1)
            df['true_range'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['prev_close']),
                    abs(df['Low'] - df['prev_close'])
                )
            )
            df['ATR_14'] = df['true_range'].rolling(window=14, min_periods=1).mean()
            df.drop(['prev_close', 'true_range'], axis=1, inplace=True)
        
        # Market structure features
        df['higher_high'] = ((df['Close'] > df['Close'].shift(1)) & 
                            (df['Close'].shift(1) > df['Close'].shift(2))).astype(int)
        df['lower_low'] = ((df['Close'] < df['Close'].shift(1)) & 
                          (df['Close'].shift(1) < df['Close'].shift(2))).astype(int)
        
        # Seasonal/Cyclical features
        if 'Date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['Date']).dt.day
            df['month'] = pd.to_datetime(df['Date']).dt.month
            df['quarter'] = pd.to_datetime(df['Date']).dt.quarter
        
        # Statistical features
        df['close_zscore_20'] = (df['Close'] - df['Close'].rolling(20, min_periods=1).mean()) / (df['Close'].rolling(20, min_periods=1).std() + 1e-10)
        df['close_percentile_20'] = df['Close'].rolling(20, min_periods=1).rank(pct=True)
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        st.write("Columns after adding enhanced indicators:", df.columns.tolist())
        
    except Exception as e:
        st.error(f"Error adding technical indicators: {str(e)}")
        
    return df

def prepare_features(df):
    """Prepare comprehensive features for machine learning"""
    try:
        df = add_technical_indicators(df)
        df = df.copy()
        
        # Enhanced lagged features (multiple time horizons)
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag{lag}'] = df['returns_1d'].shift(lag) if 'returns_1d' in df.columns else df['Close'].pct_change().shift(lag)
        
        # Rolling statistics as features
        for window in [5, 10, 20]:
            df[f'close_mean_{window}'] = df['Close'].rolling(window, min_periods=1).mean()
            df[f'close_std_{window}'] = df['Close'].rolling(window, min_periods=1).std()
            df[f'close_min_{window}'] = df['Close'].rolling(window, min_periods=1).min()
            df[f'close_max_{window}'] = df['Close'].rolling(window, min_periods=1).max()
            df[f'close_range_{window}'] = df[f'close_max_{window}'] - df[f'close_min_{window}']
        
        st.write("Columns after adding enhanced lags and rolling stats:", df.columns.tolist())

        # Define comprehensive feature set
        base_features = [
            # Lagged prices and returns
            'close_lag1', 'close_lag2', 'close_lag3', 'close_lag5', 'close_lag10',
            'returns_lag1', 'returns_lag2', 'returns_lag3',
            
            # Moving averages and ratios
            'SMA_5', 'SMA_14', 'SMA_50', 'EMA_14', 'EMA_50',
            'SMA_5_14_ratio', 'SMA_14_50_ratio', 'EMA_14_50_ratio',
            'price_above_SMA14', 'price_above_SMA50',
            
            # Volatility and momentum
            'volatility_10', 'volatility_30', 'volatility_ratio',
            'returns_1d', 'returns_3d', 'returns_7d', 'returns_14d',
            'cum_returns_5d', 'cum_returns_10d',
            
            # Technical indicators
            'RSI_7', 'RSI_14', 'RSI_21',
            'MACD', 'MACD_signal', 'MACD_histogram',
            
            # Bollinger Bands
            'BB_width_20', 'BB_position_20', 'BB_width_50', 'BB_position_50',
            
            # Statistical features
            'close_zscore_20', 'close_percentile_20',
            
            # Market structure
            'higher_high', 'lower_low',
            
            # Rolling statistics
            'close_mean_5', 'close_std_5', 'close_range_5',
            'close_mean_10', 'close_std_10', 'close_range_10',
            'close_mean_20', 'close_std_20', 'close_range_20'
        ]
        
        # Add volume features if available
        volume_features = []
        if 'Volume' in df.columns:
            volume_features = [
                'volume_sma_10', 'volume_sma_30', 'volume_ratio', 
                'volume_price_trend', 'OBV', 'OBV_sma'
            ]
        
        # Add high-low features if available
        high_low_features = []
        if 'High' in df.columns and 'Low' in df.columns:
            high_low_features = [
                'daily_range', 'daily_range_pct', 'high_low_ratio', 'ATR_14'
            ]
        
        # Add seasonal features if available
        seasonal_features = []
        if 'day_of_week' in df.columns:
            seasonal_features = ['day_of_week', 'day_of_month', 'month', 'quarter']
        
        # Combine all features
        all_features = base_features + volume_features + high_low_features + seasonal_features
        
        # Filter features that actually exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        st.write(f"Available features ({len(available_features)}): {available_features[:10]}{'...' if len(available_features) > 10 else ''}")

        # Check for missing columns
        missing_cols = [col for col in available_features + ['Close'] if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns before processing: {missing_cols}")
            return pd.DataFrame(), pd.Series(dtype=float)

        # Drop rows with NaN values
        df_clean = df.dropna(subset=available_features + ['Close'])
        st.write(f"Data shape after dropna: {df_clean.shape}")
        
        if len(df_clean) < 100:  # Increased minimum for more complex features
            st.error(f"Insufficient data after cleaning: {len(df_clean)} rows. Need at least 100 rows for enhanced features.")
            return pd.DataFrame(), pd.Series(dtype=float)

        X = df_clean[available_features]
        y = df_clean['Close']
        
        # Feature selection using correlation and variance
        st.subheader("🔍 Feature Analysis")
        
        # Remove features with very low variance
        feature_variances = X.var()
        low_variance_features = feature_variances[feature_variances < 1e-6].index.tolist()
        if low_variance_features:
            st.write(f"Removing {len(low_variance_features)} low-variance features")
            X = X.drop(columns=low_variance_features)
        
        # Remove highly correlated features
        correlation_matrix = X.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [column for column in upper_triangle.columns 
                             if any(upper_triangle[column] > 0.95)]
        if high_corr_features:
            st.write(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
        
        # Feature importance preview (quick RF)
        if len(X) > 50 and len(X.columns) > 5:
            try:
                from sklearn.ensemble import RandomForestRegressor
                quick_rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                split_idx = int(len(X) * 0.8)
                X_temp_train = X.iloc[:split_idx]
                y_temp_train = y.iloc[:split_idx]
                quick_rf.fit(X_temp_train, y_temp_train)
                
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': quick_rf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Keep top features for performance
                top_features = feature_importance.head(min(30, len(X.columns)))['feature'].tolist()
                X = X[top_features]
                st.write(f"Selected top {len(top_features)} most important features")
                
            except Exception as e:
                st.write(f"Feature selection failed, using all features: {e}")
        
        st.write(f"Final feature set: {len(X.columns)} features")
        st.write(f"Final data shape: {X.shape}")
        
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
    """Train multiple models and select the best one with comprehensive evaluation"""
    try:
        # Use temporal split (no shuffling for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        st.write(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                random_state=42,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt'
            ),
            'Random Forest (Deep)': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.8
            ),
            'Random Forest (Wide)': RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt'
            )
        }
        
        # If we have enough features, try gradient boosting
        if len(X.columns) >= 10:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                models['Gradient Boosting'] = GradientBoostingRegressor(
                    n_estimators=150,
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8
                )
            except ImportError:
                pass
        
        best_model = None
        best_score = -np.inf
        best_metrics = None
        model_results = {}
        
        st.subheader("🤖 Model Comparison")
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                # Calculate metrics
                test_r2 = r2_score(y_test, test_preds)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
                test_mae = mean_absolute_error(y_test, test_preds)
                
                model_results[name] = {
                    'r2': test_r2,
                    'rmse': test_rmse,
                    'mae': test_mae
                }
                
                # Update best model based on R2 score
                if test_r2 > best_score:
                    best_score = test_r2
                    best_model = model
                    best_name = name
                    
                    # Store comprehensive metrics for best model
                    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
                    train_mae = mean_absolute_error(y_train, train_preds)
                    train_r2 = r2_score(y_train, train_preds)
                    train_mape = np.mean(np.abs((y_train - train_preds) / y_train)) * 100
                    test_mape = np.mean(np.abs((y_test - test_preds) / y_test)) * 100
                    
                    best_metrics = {
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
                
            except Exception as e:
                st.write(f"Failed to train {name}: {e}")
                continue
        
        # Display model comparison
        if model_results:
            comparison_df = pd.DataFrame(model_results).T
            comparison_df = comparison_df.round(4)
            st.write("**Model Performance Comparison:**")
            st.dataframe(comparison_df)
            
            st.success(f"🏆 Best Model: **{best_name}** (R² = {best_score:.4f})")
        
        if best_model is None:
            st.error("All models failed to train")
            return None, None
        
        # Display comprehensive metrics for best model
        st.subheader("📊 Best Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Set Performance:**")
            st.metric("RMSE", f"${best_metrics['train']['rmse']:.2f}")
            st.metric("MAE", f"${best_metrics['train']['mae']:.2f}")
            st.metric("R² Score", f"{best_metrics['train']['r2']:.4f}")
            st.metric("MAPE", f"{best_metrics['train']['mape']:.2f}%")
        
        with col2:
            st.write("**Test Set Performance:**")
            st.metric("RMSE", f"${best_metrics['test']['rmse']:.2f}")
            st.metric("MAE", f"${best_metrics['test']['mae']:.2f}")
            st.metric("R² Score", f"{best_metrics['test']['r2']:.4f}")
            st.metric("MAPE", f"{best_metrics['test']['mape']:.2f}%")
        
        # Performance interpretation with more nuanced thresholds
        test_r2 = best_metrics['test']['r2']
        if test_r2 > 0.7:
            performance_msg = "🟢 Excellent model performance!"
        elif test_r2 > 0.5:
            performance_msg = "🟡 Good model performance"
        elif test_r2 > 0.3:
            performance_msg = "🟠 Moderate model performance"
        elif test_r2 > 0.1:
            performance_msg = "🔶 Fair model performance - room for improvement"
        else:
            performance_msg = "🔴 Poor model performance - consider feature engineering"
        
        st.info(f"**Model Assessment:** {performance_msg}")
        
        # Enhanced improvement suggestions
        if test_r2 < 0.5:
            st.subheader("💡 Specific Improvement Strategies")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔧 Feature Engineering:**")
                if test_r2 < 0.2:
                    st.write("- ✅ Add more lagged features (3, 5, 10 days)")
                    st.write("- ✅ Try different MA periods (5, 21, 50, 100)")
                    st.write("- ✅ Add momentum indicators (ROC, Stochastic)")
                    st.write("- ✅ Include market volatility (VIX if available)")
                    st.write("- ✅ Add price transformation (log returns)")
                
                if len(X.columns) < 25:
                    st.write("- 📊 Enable volume features")
                    st.write("- 📅 Enable seasonal features") 
                    st.write("- 🎯 Add support/resistance levels")
                
            with col2:
                st.write("**📈 Data & Model:**")
                st.write("- ⏰ Try longer data period (2y or 5y)")
                st.write("- 🎭 Test different prediction horizons")
                st.write("- 🔄 Use different train/test splits")
                st.write("- 🎪 Try ensemble methods")
                
                mape = best_metrics['test']['mape']
                if mape > 15:
                    st.write("- 🎯 Consider predicting direction instead of price")
                    st.write("- 🔍 Check for regime changes in data")
                    st.write("- 📊 Try different target transformations")
            
            # Stock-specific advice
            st.write("**📊 Stock-Specific Tips:**")
            current_volatility = best_metrics['test']['rmse'] / np.mean(best_metrics['test']['actual'])
            if current_volatility > 0.1:
                st.warning("⚠️ High volatility stock - consider shorter prediction horizons")
            else:
                st.info("ℹ️ Stable stock - try longer-term predictions")
            
            # Advanced techniques suggestion
            if test_r2 < 0.3:
                st.error("🔬 **Advanced Techniques Needed:**")
                st.write("- Try LSTM/GRU neural networks")
                st.write("- Use ensemble of different model types")  
                st.write("- Consider external factors (news, sentiment)")
                st.write("- Implement regime-switching models")
                st.write("- Use cross-validation with time series splits")
        
        # Check for overfitting with more detail
        r2_diff = best_metrics['train']['r2'] - best_metrics['test']['r2']
        if r2_diff > 0.2:
            st.warning(f"⚠️ Significant overfitting detected (Training R²: {best_metrics['train']['r2']:.3f}, Test R²: {best_metrics['test']['r2']:.3f})")
            st.write("**Overfitting Solutions:**")
            st.write("- Reduce model complexity (fewer trees, lower depth)")
            st.write("- Increase regularization")
            st.write("- Use more training data")
            st.write("- Apply feature selection")
        elif r2_diff > 0.1:
            st.warning(f"⚠️ Moderate overfitting detected (Training R²: {best_metrics['train']['r2']:.3f}, Test R²: {best_metrics['test']['r2']:.3f})")
        elif r2_diff < 0.05:
            st.success("✅ Model shows good generalization (low overfitting)")
        
        return best_model, best_metrics
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None(X) * 0.8)
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
if st.button("Fetch, Train & Predict", type="primary"):
    with st.spinner("Fetching data..."):
        data = fetch_data(ticker, period=data_period)
        
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

                # Make final prediction
                st.subheader("🎯 Price Prediction")
                last_features = X.iloc[-1].values
                predicted_price = predict_next(model, last_features)
                
                if predicted_price is not None:
                    current_price = data['Close'].iloc[-1]
                    price_change = predicted_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    # Display prediction with confidence context
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        st.metric(
                            "Predicted Next Price", 
                            f"${predicted_price:.2f}",
                            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                        )
                    
                    with col3:
                        # Confidence level based on model performance
                        confidence_level = "High" if metrics['test']['r2'] > 0.7 else "Medium" if metrics['test']['r2'] > 0.4 else "Low"
                        confidence_color = "green" if confidence_level == "High" else "orange" if confidence_level == "Medium" else "red"
                        st.markdown(f"**Confidence:** :{confidence_color}[{confidence_level}]")
                        
                        # Expected error range
                        expected_error = metrics['test']['mae']
                        st.write(f"Expected error: ±${expected_error:.2f}")
                    
                    # Prediction context
                    if abs(price_change_pct) > 5:
                        st.warning(f"⚠️ Large predicted change ({price_change_pct:+.1f}%) - use with caution")
                    elif abs(price_change_pct) < 1:
                        st.info("ℹ️ Small predicted change - market may be stable")
                    else:
                        st.success("✅ Moderate predicted change within normal range")

                    # Create final visualization with historical data and prediction
                    st.subheader("📊 Price History and Prediction")
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Plot 1: Full historical prices and prediction
                    ax1.plot(data['Date'], data['Close'], label='Historical Close', linewidth=1.5, color='blue')
                    
                    # Add prediction point
                    last_date = data['Date'].iloc[-1]
                    if hasattr(last_date, 'to_pydatetime'):
                        predicted_date = last_date.to_pydatetime() + timedelta(days=1)
                    else:
                        predicted_date = last_date + timedelta(days=1)
                    
                    ax1.scatter(predicted_date, predicted_price, color='red', 
                              label=f'Predicted Close (${predicted_price:.2f})', 
                              s=150, marker='*', zorder=5, edgecolors='black', linewidth=1)
                    
                    # Add confidence interval around prediction
                    expected_error = metrics['test']['mae']
                    ax1.fill_between([predicted_date, predicted_date], 
                                    [predicted_price - expected_error, predicted_price - expected_error],
                                    [predicted_price + expected_error, predicted_price + expected_error],
                                    alpha=0.2, color='red', label=f'±{expected_error:.2f} confidence')
                    
                    ax1.set_title(f"{ticker} Full Price History and Prediction")
                    ax1.set_xlabel("Date")
                    ax1.set_ylabel("Price ($)")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Recent performance focus with model performance overlay
                    recent_data = data.tail(60)  # Last 60 days
                    ax2.plot(recent_data['Date'], recent_data['Close'], 
                            label='Recent Close', linewidth=2, color='blue')
                    ax2.scatter(predicted_date, predicted_price, color='red', 
                              label=f'Predicted (R²={metrics["test"]["r2"]:.3f})', 
                              s=150, marker='*', zorder=5, edgecolors='black', linewidth=1)
                    
                    # Add error bars for prediction
                    ax2.errorbar(predicted_date, predicted_price, yerr=expected_error,
                               fmt='none', color='red', capsize=5, capthick=2, alpha=0.7)
                    
                    ax2.set_title(f"Recent Performance (Last 60 Days) - Model RMSE: ${metrics['test']['rmse']:.2f}")
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
This app predicts stock prices using machine learning with enhanced features.

**Enhanced Features:**
- Multiple moving averages & crossovers
- Volatility indicators
- Momentum & trend features  
- Multiple RSI periods
- Enhanced MACD with signals
- Bollinger Bands positioning
- Volume analysis (when available)
- Statistical features
- Market structure patterns
- Seasonal components

**Models:** Automatic selection from:
- Random Forest (multiple configurations)
- Gradient Boosting (when available)

**Note:** This is for educational purposes only. 
Do not use for actual trading decisions.
""")

# Model Configuration
st.sidebar.header("⚙️ Configuration")

# Data settings
st.sidebar.subheader("Data Settings")
data_period = st.sidebar.selectbox(
    "Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=1,
    help="Longer periods provide more training data but may include outdated patterns"
)

# Feature engineering options
st.sidebar.subheader("Feature Engineering")
use_volume = st.sidebar.checkbox("Use Volume Features", value=True, help="Include volume-based indicators when available")
use_seasonal = st.sidebar.checkbox("Use Seasonal Features", value=True, help="Include day/month/quarter features")
max_features = st.sidebar.slider("Max Features", 10, 50, 30, help="Maximum number of features to use")

# Model settings
st.sidebar.subheader("Model Settings")
ensemble_size = st.sidebar.selectbox(
    "Model Ensemble",
    ["Single Best", "Top 2 Average", "All Models Average"],
    help="How to combine model predictions"
)

prediction_confidence = st.sidebar.selectbox(
    "Prediction Confidence",
    ["Conservative", "Moderate", "Aggressive"],
    index=1,
    help="How to interpret model confidence levels"
)

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

**For Poor Model Performance:**
- Try longer data periods (1y → 2y)
- Use popular, liquid stocks (AAPL, MSFT, GOOGL)
- Enable all feature options
- Check if stock has consistent patterns
- Volatile or trending markets work better than sideways markets
""")

# Model Performance Guide
st.sidebar.header("📈 Performance Guide")
st.sidebar.info("""
**R² Score Interpretation:**
- 0.7+ = Excellent (rare for stock prediction)
- 0.5-0.7 = Good  
- 0.3-0.5 = Moderate
- 0.1-0.3 = Fair
- <0.1 = Poor

**Improvement Tips:**
- **More data**: Use 2y+ for better patterns
- **Different stocks**: Try trending/volatile stocks
- **Feature engineering**: Enable all options
- **Market conditions**: Bull/bear markets easier to predict than sideways
""")

# Quick Performance Test
st.sidebar.header("🚀 Quick Test")
if st.sidebar.button("Test with AAPL (2Y)", help="Quick test with known good parameters"):
    st.session_state.ticker_input = "AAPL"
    st.session_state.test_mode = True
    st.rerun()

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
