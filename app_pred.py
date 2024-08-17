import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from scipy.stats import skew, kurtosis
import streamlit as st

st.title("Stocks Predective Analytics Dashboard")
st.text("Prepard by Rami Aldoush")
# Function to fetch stock data
def fetch_stock_data(stock_name, start_year):
    end_date = datetime(2024, 8, 31)
    return yf.download(stock_name, start=start_year, end=end_date)

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data['MA50'] = ta.sma(data['Close'], length=50)
    data['MA200'] = ta.sma(data['Close'], length=200)
    data['RSI'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data = pd.concat([data, macd], axis=1)
    data['MACD'] = data['MACD_12_26_9']
    data['MACD_signal'] = data['MACDs_12_26_9']
    data.dropna(inplace=True)

    # Add MACD Buy/Sell signals
    data['MACD_buy_signal'] = (data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1))
    data['MACD_sell_signal'] = (data['MACD'] < data['MACD_signal']) & (data['MACD'].shift(1) >= data['MACD_signal'].shift(1))

    return data

# Function to predict stock prices
def predict_stock_price(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()

    forecast_days = 126  # approximately 6 months
    forecast = model_fit.get_forecast(steps=forecast_days)
    forecast_index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_days, freq='B')

    forecast_series = forecast.predicted_mean
    conf_int = forecast.conf_int()

    future_dates = forecast_index
    future_predictions = forecast_series

    return future_dates, future_predictions, conf_int.values

# Function to calculate and display advanced statistics
def calculate_statistics(data):
    mean = data.mean()
    variance = data.var()
    std_dev = data.std()
    skewness = skew(data)
    kurt = kurtosis(data)
    
    return mean, variance, std_dev, skewness, kurt

# Function to analyze buying opportunity
def analyze_buying_opportunity(data):
    latest_data = data.iloc[-1]
    
    if latest_data['RSI'] < 30:
        rsi_signal = "RSI indicates the stock is oversold. Buying opportunity."
    elif latest_data['RSI'] > 70:
        rsi_signal = "RSI indicates the stock is overbought. Selling opportunity."
    else:
        rsi_signal = "RSI indicates the stock is neither overbought nor oversold."

    if latest_data['MACD'] > latest_data['MACD_signal']:
        macd_signal = "MACD is above the signal line. Bullish signal."
    else:
        macd_signal = "MACD is below the signal line. Bearish signal."

    st.write(f"RSI Analysis: {rsi_signal}")
    st.write(f"MACD Analysis: {macd_signal}")

# Main function for Streamlit app
def main():
    stock_data = fetch_stock_data(stock_name, start_year)
    
    if stock_data.empty:
        st.write("Failed to fetch data. Please check the stock ticker and try again.")
        return

    stock_data = calculate_technical_indicators(stock_data)
    
    future_dates, future_predictions, conf_int = predict_stock_price(stock_data)
    
    # Display gain/loss calculation
    initial_price = stock_data.iloc[0]['Close']
    latest_price = stock_data.iloc[-1]['Close']
    gain_loss = latest_price - initial_price
    percent_change = (gain_loss / initial_price) * 100
    st.write(f"If you bought {stock_name} at the beginning of {start_year.year} at ${initial_price:.2f},")
    st.write(f"the latest price would be ${latest_price:.2f}.")
    st.write(f"Your gain/loss would be: ${gain_loss:.2f}, which is a {percent_change:.2f}% change.")

    # 1. Closing Prices with Moving Averages
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price', line=dict(color='blue', dash='dash')))
    fig_close.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], mode='lines', name='50-Day MA', line=dict(color='red')))
    fig_close.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], mode='lines', name='200-Day MA', line=dict(color='green')))
    fig_close.update_layout(title='Stock Price and Moving Averages', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    st.plotly_chart(fig_close)

    # 2. RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line=dict(color='red', dash='dash'))
    fig_rsi.add_hline(y=30, line=dict(color='green', dash='dash'))
    fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI', template='plotly_white')
    
    # Add RSI statistics
    rsi_mean, rsi_variance, rsi_std_dev, rsi_skewness, rsi_kurt = calculate_statistics(stock_data['RSI'])
    st.write(f"RSI Mean: {rsi_mean:.2f}")
    st.write(f"RSI Variance: {rsi_variance:.2f}")
    st.write(f"RSI Standard Deviation: {rsi_std_dev:.2f}")
    st.write(f"RSI Skewness: {rsi_skewness:.2f}")
    st.write(f"RSI Kurtosis: {rsi_kurt:.2f}")
    
    st.plotly_chart(fig_rsi)

    # 3. MACD with Buy/Sell Signals
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_signal'], mode='lines', name='Signal Line', line=dict(color='red')))
    fig_macd.add_trace(go.Scatter(x=stock_data[stock_data['MACD_buy_signal']].index, y=stock_data['MACD'][stock_data['MACD_buy_signal']], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up')))
    fig_macd.add_trace(go.Scatter(x=stock_data[stock_data['MACD_sell_signal']].index, y=stock_data['MACD'][stock_data['MACD_sell_signal']], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down')))
    fig_macd.update_layout(title='MACD (Moving Average Convergence Divergence)', xaxis_title='Date', yaxis_title='MACD', template='plotly_white')
    
    # Add MACD statistics
    macd_mean, macd_variance, macd_std_dev, macd_skewness, macd_kurt = calculate_statistics(stock_data['MACD'])
    st.write(f"MACD Mean: {macd_mean:.2f}")
    st.write(f"MACD Variance: {macd_variance:.2f}")
    st.write(f"MACD Standard Deviation: {macd_std_dev:.2f}")
    st.write(f"MACD Skewness: {macd_skewness:.2f}")
    st.write(f"MACD Kurtosis: {macd_kurt:.2f}")
    
    st.plotly_chart(fig_macd)

    # 4. Historical and Forecasted Prices
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Price', line=dict(color='blue', dash='dash')))
    fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Forecasted Price', line=dict(color='orange')))
    fig_forecast.add_trace(go.Scatter(x=future_dates, y=conf_int[:, 0], fill=None, mode='lines', line=dict(color='orange', dash='dash')))
    fig_forecast.add_trace(go.Scatter(x=future_dates, y=conf_int[:, 1], fill='tonexty', mode='lines', line=dict(color='orange', dash='dash')))
    fig_forecast.update_layout(title='Historical and Forecasted Prices', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    
    # Add forecasting statistics
    forecast_mean, forecast_variance, forecast_std_dev, forecast_skewness, forecast_kurt = calculate_statistics(pd.Series(future_predictions))
    st.write(f"Forecast Mean: {forecast_mean:.2f}")
    st.write(f"Forecast Variance: {forecast_variance:.2f}")
    st.write(f"Forecast Standard Deviation: {forecast_std_dev:.2f}")
    st.write(f"Forecast Skewness: {forecast_skewness:.2f}")
    st.write(f"Forecast Kurtosis: {forecast_kurt:.2f}")
    
    st.plotly_chart(fig_forecast)

    # Analyze buying opportunity
    analyze_buying_opportunity(stock_data)
    
# Sidebar for User Input
st.sidebar.title('Stock Analysis Parameters')
stock_name = st.sidebar.text_input('Enter the stock ticker (e.g., AAPL):', 'AAPL')
start_year = st.sidebar.date_input('Select start year for analysis:', datetime(2021, 1, 1))

if __name__ == "__main__":
    main()
