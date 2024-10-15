import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import pandas_ta as ta  # Import pandas-ta for technical indicators
from sklearn.preprocessing import StandardScaler  # For standardization

# List of tickers for the six companies
tickers = ["LLY", "ABBV", "JNJ", "MRK", "TMO", "UNH"]

# Define the start and end dates
start_date = '2014-09-01'
end_date = '2024-09-01'

# Loop through each ticker, download the data, process it, and save to CSV
for ticker in tickers:
    print(f"Processing data for {ticker}...")

    # Download historical stock data using yf.download() to get Date, Open, High, Low, Close, Adj Close, Volume
    hist = yf.download(ticker, start=start_date, end=end_date)

    # Save the raw data to a CSV file before any processing
    raw_csv_filename = f"{ticker}_raw_data.csv"
    hist.to_csv(raw_csv_filename)
    print(f"Raw data for {ticker} saved to '{raw_csv_filename}'")

    # Check for missing data and handle it
    if hist.isnull().values.any():
        print(f"Missing data found for {ticker}. Handling missing data...")
        hist = hist.fillna(method='ffill')
    else:
        print(f"No missing data found for {ticker}.")

    # Feature Engineering using pandas-ta

    # Moving Averages
    hist['5_day_SMA'] = hist['Close'].rolling(window=5).mean()
    hist['20_day_SMA'] = hist['Close'].rolling(window=20).mean()
    hist['50_day_SMA'] = hist['Close'].rolling(window=50).mean()

    # RSI (Relative Strength Index) using pandas-ta
    hist['14_day_RSI'] = ta.rsi(hist['Close'], length=14)

    # Lagged Close (1 day)
    hist['Lagged_Close'] = hist['Close'].shift(1)

    # ATR (Average True Range) using pandas-ta
    hist['ATR'] = ta.atr(hist['High'], hist['Low'], hist['Close'], length=14)

    # Days of the Week as a feature
    hist['Day_of_Week'] = hist.index.dayofweek  # 0 = Monday, 6 = Sunday

    # Daily Returns
    hist['Daily_Return'] = hist['Close'].pct_change()

    # Volatility (Standard Deviation of Daily Returns over 5 days)
    hist['Volatility'] = hist['Daily_Return'].rolling(window=5).std()

    # Standardization section (commented out)
    # Uncomment these lines if you want to apply standardization to the features
    # scaler = StandardScaler()
    # features_to_standardize = ['5_day_SMA', '20_day_SMA', '50_day_SMA',
    #                            '14_day_RSI', 'Lagged_Close', 'ATR',
    #                            'Day_of_Week', 'Daily_Return', 'Volatility']

    # Check for non-empty features and fill NaNs (in case of missing rolling window calculations)
    # hist[features_to_standardize] = hist[features_to_standardize].fillna(0)

    # Apply standardization
    # hist[features_to_standardize] = scaler.fit_transform(hist[features_to_standardize])

    # Save the processed data to a CSV file named after the company ticker
    processed_csv_filename = f"{ticker}_processed_data.csv"
    hist.to_csv(processed_csv_filename)

    print(f"Processed data for {ticker} saved to '{processed_csv_filename}'.")

print("Data processing for all tickers completed.")

# trying my github