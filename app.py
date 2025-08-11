import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

st.title("Algorithmic Trading Price Predictor")

ticker = st.text_input("Enter ticker symbol", value="AAPL").upper()

@st.cache_data(ttl=3600)
def fetch_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    return df.reset_index() if not df.empty else pd.DataFrame()

def prepare_features(df):
    df['close_lag1'] = df['Close'].shift(1)
    df['close_lag2'] = df['Close'].shift(2)
    df = df.dropna()
    X = df[['close_lag1', 'close_lag2']]
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

def predict_next(model, last_two_closes):
    pred = model.predict(np.array(last_two_closes).reshape(1, -1))
    return pred[0]

if st.button("Fetch, Train & Predict"):
    with st.spinner("Fetching data..."):
        data = fetch_data(ticker)
    if data.empty:
        st.error("No data found for ticker. Try another symbol.")
    else:
        st.write(f"Fetched {len(data)} rows for {ticker}")

        X, y = prepare_features(data)
        if len(X) == 0:
            st.error("Not enough data to prepare features.")
        else:
            st.write("Training model...")
            model, rmse = train_model(X, y)
            st.write(f"Model trained. RMSE on test set: {rmse:.2f}")

            last_two = data['Close'].iloc[-2:].values
            predicted_price = predict_next(model, last_two)
            st.success(f"Predicted next closing price for {ticker}: ${predicted_price:.2f}")

            # Plot historical close prices
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data['Date'], data['Close'], label='Historical Close', color='blue')

            # Predicted date is one trading day after last date
            last_date = data['Date'].iloc[-1]
            predicted_date = last_date + timedelta(days=1)

            # Plot predicted price as a red star
            ax.scatter(predicted_date, predicted_price, color='red', label='Predicted Close', s=100, marker='*')

            ax.set_title(f"{ticker} Closing Prices and Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
