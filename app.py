import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta

st.title("Algorithmic Trading Price Predictor - Debug Version")

ticker = st.text_input("Enter ticker symbol", "AAPL").upper()

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    st.write(f"Raw data columns from yfinance: {df.columns.tolist()}")
    return df.reset_index() if not df.empty else pd.DataFrame()

def add_technical_indicators(df):
    df = df.copy()
    st.write("Columns at start of add_technical_indicators:", df.columns.tolist())

    if 'Close' not in df.columns:
        st.error("ERROR: 'Close' column missing from data.")
        return df

    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = sma_20 + (std_20 * 2)
    df['Bollinger_Lower'] = sma_20 - (std_20 * 2)

    st.write("Columns after adding indicators:", df.columns.tolist())
    return df

def prepare_features(df):
    df = add_technical_indicators(df)
    df = df.copy()
    df['close_lag1'] = df['Close'].shift(1)
    df['close_lag2'] = df['Close'].shift(2)

    st.write("Columns after adding lags:", df.columns.tolist())

    features = ['close_lag1', 'close_lag2', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']

    missing_cols = [col for col in features + ['Close'] if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns before dropna: {missing_cols}")
        return pd.DataFrame(), pd.Series(dtype=float)

    df = df.dropna(subset=features + ['Close'])
    st.write(f"Data shape after dropna: {df.shape}")

    X = df[features]
    y = df['Close']
    return X, y

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, rmse

def predict_next(model, last_features):
    pred = model.predict(np.array(last_features).reshape(1, -1))
    return pred[0]

if st.button("Fetch, Train & Predict"):
    data = fetch_data(ticker)
    if data.empty:
        st.error("No data found for ticker.")
    else:
        st.write(f"Fetched {len(data)} rows for {ticker}")
        X, y = prepare_features(data)
        if len(X) == 0:
            st.error("Not enough data or missing columns for features.")
        else:
            model, rmse = train_model(X, y)
            st.write(f"Model trained. RMSE on test set: {rmse:.2f}")

            last_features = X.iloc[-1].values
            predicted_price = predict_next(model, last_features)
            st.success(f"Predicted next closing price for {ticker}: ${predicted_price:.2f}")

            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(data['Date'], data['Close'], label='Historical Close')
            last_date = data['Date'].iloc[-1]
            predicted_date = last_date + timedelta(days=1)
            ax.scatter(predicted_date, predicted_price, color='red', label='Predicted Close', s=100, marker='*')
            ax.set_title(f"{ticker} Closing Prices and Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
