import yfinance as yf
import pandas as pd
import numpy as np
import os  # For file path management
import pandas_ta as ta  # Import pandas-ta for technical indicators
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For evaluating forecast
from statsmodels.tsa.stattools import adfuller  # Import ADF test
from arch.unitroot import DFGLS  # Import DFGLS test (corrected)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For ACF and PACF plotting
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA for modeling
import matplotlib.pyplot as plt  # For plotting

# List of tickers for the six companies
tickers = ["LLY", "ABBV", "JNJ", "MRK", "TMO", "UNH"]

# Define the start and end dates
start_date = '2014-09-01'
end_date = '2024-09-01'

# Define the folder path for saving CSV files
data_folder = os.path.join(os.getcwd(), 'data')

# Create the folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Function to check stationarity using ADF and DFGLS
def check_stationarity(series, name=""):
    # ADF Test
    adf_result = adfuller(series)
    print(f"ADF Test Statistic for {name}: {adf_result[0]}")
    print(f"ADF p-value for {name}: {adf_result[1]}")
    if adf_result[1] <= 0.05:
        print(f"The series '{name}' is stationary based on the ADF test (p <= 0.05).")
    else:
        print(f"The series '{name}' is NOT stationary based on the ADF test (p > 0.05).")

    # DFGLS Test
    dfgls_result = DFGLS(series).stat
    print(f"DFGLS Test Statistic for {name}: {dfgls_result}")
    if dfgls_result < -2.86:
        print(f"The series '{name}' is stationary based on the DFGLS test.")
    else:
        print(f"The series '{name}' is NOT stationary based on the DFGLS test.")


# Loop through each ticker, download the data, process it, and save to CSV
for ticker in tickers:
    print(f"\nProcessing data for {ticker}...")

    # Download historical stock data using yf.download() to get Date, Open, High, Low, Close, Adj Close, Volume
    hist = yf.download(ticker, start=start_date, end=end_date)

    # Save the raw data to a CSV file before any processing
    raw_csv_filename = os.path.join(data_folder, f"{ticker}_raw_data.csv")
    hist.to_csv(raw_csv_filename)
    print(f"Raw data for {ticker} saved to '{raw_csv_filename}'")

    # Check for missing data and handle it
    if hist.isnull().values.any():
        print(f"Missing data found for {ticker}. Handling missing data...")
        hist = hist.fillna(method='ffill')
    else:
        print(f"No missing data found for {ticker}.")

    # Check stationarity on the 'Close' price
    print(f"Checking stationarity for {ticker} 'Close' prices:")
    check_stationarity(hist['Close'], name="Close")

    # If the series is not stationary, apply differencing
    print(f"Applying differencing for {ticker} 'Close' prices...")
    hist['Close_diff'] = hist['Close'].diff().dropna()  # Apply first-order differencing and drop NaN values

    # Check stationarity again after differencing (ensure no NaNs are present)
    print(f"Checking stationarity for differenced {ticker} 'Close' prices:")
    check_stationarity(hist['Close_diff'].dropna(), name="Close_diff")  # Drop NaNs before checking stationarity
    print(f"Summary of original 'Close' data for {ticker}:")
    print(hist['Close'].describe())
    print(f"\nSummary of differenced 'Close_diff' data for {ticker}:")
    print(hist['Close_diff'].describe())

    # Plot ACF and PACF for the differenced series
    print(f"\nPlotting ACF and PACF for {ticker} 'Close_diff'...")
    plt.figure(figsize=(12, 6))

    # ACF Plot
    plt.subplot(121)
    plot_acf(hist['Close_diff'].dropna(), lags=40, ax=plt.gca())
    plt.title(f"ACF for {ticker} 'Close_diff'")

    # PACF Plot
    plt.subplot(122)
    plot_pacf(hist['Close_diff'].dropna(), lags=40, ax=plt.gca())
    plt.title(f"PACF for {ticker} 'Close_diff'")

    plt.tight_layout()
    plt.show()

    # Set date index frequency before fitting ARIMA
    hist.index = pd.to_datetime(hist.index)
    hist = hist.asfreq('B')  # 'B' stands for business days

    # Train-test split (80% train, 20% test)
    split_point = int(len(hist) * 0.8)
    train_data = hist['Close'][:split_point]
    test_data = hist['Close'][split_point:]

    # Drop NaN values from train and test sets (if any)
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Fit ARIMA model on training data (p=1, d=1, q=1 based on the plots)
    print(f"Fitting ARIMA model for {ticker} (p=1, d=1, q=1) on training data...")
    model = ARIMA(train_data, order=(1, 1, 1))
    arima_result = model.fit()

    # Print ARIMA model summary
    print(arima_result.summary())

    # Forecast the values for the test set
    forecast = arima_result.forecast(steps=len(test_data))

    # Drop NaN values in the forecast and test_data (to align shapes)
    forecast = forecast.dropna()
    test_data = test_data[:len(forecast)]

    # Plot actual vs forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data, label='Actual Close Prices', color='blue')
    plt.plot(test_data.index, forecast, label='Forecasted Close Prices', color='red')
    plt.title(f'ARIMA Forecast vs Actual for {ticker}')
    plt.legend()
    plt.show()

    # Evaluate the model performance
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    print(f"Mean Absolute Error (MAE) for {ticker}: {mae}")
    print(f"Root Mean Squared Error (RMSE) for {ticker}: {rmse}")

    # Plot the residuals to check the fit
    residuals = arima_result.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title(f'Residuals of ARIMA model for {ticker}')
    plt.show()

    # Plot ACF of residuals to check if residuals are white noise
    print(f"Plotting ACF of residuals for {ticker}...")
    plot_acf(residuals)
    plt.show()

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

    # Save the processed data to a CSV file named after the company ticker
    processed_csv_filename = os.path.join(data_folder, f"{ticker}_processed_data.csv")
    hist.to_csv(processed_csv_filename)

    print(f"Processed data for {ticker} saved to '{processed_csv_filename}'.")

print("Data processing, ARIMA modeling, and ACF/PACF plotting for all tickers completed.")
