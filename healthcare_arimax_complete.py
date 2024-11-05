import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import pandas_ta as ta

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


# Function to check stationarity using ADF test and apply differencing
def make_stationary(series, name="", max_diff_order=3):
    differencing_order = 0
    while differencing_order <= max_diff_order:
        adf_result = adfuller(series)
        if adf_result[1] <= 0.05:
            return series
        elif differencing_order == max_diff_order:
            return series
        else:
            series = series.diff().dropna()
            differencing_order += 1


# Initialize lists to store RMSE and MAE for each ticker
mae_list = []
rmse_list = []

# For each ticker, process data
for ticker in tickers:
    print(f"\nProcessing data for {ticker}...")

    # Download historical stock data
    hist = yf.download(ticker, start=start_date, end=end_date)

    # Save the raw data to CSV
    raw_csv_filename = os.path.join(data_folder, f"{ticker}_raw_data.csv")
    hist.to_csv(raw_csv_filename)

    # Ensure index is a DateTimeIndex and set frequency
    hist.index = pd.to_datetime(hist.index)
    hist = hist.asfreq('B')
    hist.ffill(inplace=True)

    # Additional Exogenous Variables
    hist['5_day_SMA'] = hist['Close'].rolling(window=5).mean()
    hist['20_day_SMA'] = hist['Close'].rolling(window=20).mean()
    hist['50_day_SMA'] = hist['Close'].rolling(window=50).mean()
    hist['14_day_RSI'] = ta.rsi(hist['Close'], window=14)
    hist['Lagged_Close'] = hist['Close'].shift(1)
    hist['ATR'] = ta.atr(hist['High'], hist['Low'], hist['Close'], window=14)
    hist['Day_of_Week'] = hist.index.dayofweek
    hist['Daily_Return'] = hist['Close'].pct_change()
    hist['Volatility'] = hist['Daily_Return'].rolling(window=5).std()

    # Handle NaN and Inf values in the newly created variables
    hist.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace Inf values with NaN
    hist.dropna(inplace=True)  # Drop rows with any NaN values

    # Apply stationarity check and differencing
    hist['Open_stationary'] = make_stationary(hist['Open'], name="Open")
    hist['Low_stationary'] = make_stationary(hist['Low'], name="Low")
    hist['High_stationary'] = make_stationary(hist['High'], name="High")
    hist['Adj_Close_stationary'] = make_stationary(hist['Adj Close'], name="Adj Close")
    hist['5_day_SMA_stationary'] = make_stationary(hist['5_day_SMA'], name="5_day_SMA")
    hist['20_day_SMA_stationary'] = make_stationary(hist['20_day_SMA'], name="20_day_SMA")
    hist['50_day_SMA_stationary'] = make_stationary(hist['50_day_SMA'], name="50_day_SMA")
    hist['14_day_RSI_stationary'] = make_stationary(hist['14_day_RSI'], name="14_day_RSI")
    hist['Lagged_Close_stationary'] = make_stationary(hist['Lagged_Close'], name="Lagged_Close")
    hist['ATR_stationary'] = make_stationary(hist['ATR'], name="ATR")
    hist['Daily_Return_stationary'] = make_stationary(hist['Daily_Return'], name="Daily_Return")
    hist['Volatility_stationary'] = make_stationary(hist['Volatility'], name="Volatility")

    # Generate lagged versions of exogenous variables
    exogenous_vars = ['Open_stationary', 'Low_stationary', 'High_stationary', 'Adj_Close_stationary',
                      '5_day_SMA_stationary', '20_day_SMA_stationary',
                      '50_day_SMA_stationary', '14_day_RSI_stationary', 'Lagged_Close_stationary', 'ATR_stationary',
                      'Daily_Return_stationary',
                      'Volatility_stationary', 'Day_of_Week']
    for var in exogenous_vars:
        hist[f"{var}_lag1"] = hist[var].shift(1)

    # Final dropna to ensure no NaN or Inf values remain
    hist.dropna(inplace=True)

    # Train-test split (80% train, 20% test)
    split_point = int(len(hist) * 0.8)
    train_data = hist[:split_point]
    test_data = hist[split_point:]

    # Define the endogenous and exogenous variables
    endog_train = train_data['Close']
    exog_train = train_data[[f"{var}_lag1" for var in exogenous_vars]]
    endog_test = test_data['Close']
    exog_test = test_data[[f"{var}_lag1" for var in exogenous_vars]]

    # Auto ARIMA parameter tuning for each ticker
    auto_arima_model = auto_arima(endog_train, exogenous=exog_train, seasonal=False, trace=True, error_action='ignore',
                                  suppress_warnings=True)
    best_order = auto_arima_model.order

    # Fit ARIMAX model with tuned parameters
    try:
        model = ARIMA(endog_train, order=best_order, exog=exog_train)
        arimax_result = model.fit()
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        try:
            simpler_order = (1, 1, 1)
            model = ARIMA(endog_train, order=simpler_order, exog=exog_train)
            arimax_result = model.fit()
        except Exception as final_e:
            continue

    # Forecast
    forecast = arimax_result.forecast(steps=len(endog_test), exog=exog_test)

    # Plot actual vs forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(endog_test.index, test_data['Close'], label='Actual Close Prices', color='blue')
    plt.plot(endog_test.index, forecast, label='Forecasted Close Prices', color='red')
    plt.title(f'ARIMAX Forecast vs Actual for {ticker}')
    plt.legend()
    plt.show()

    # Calculate MAE and RMSE
    mae = mean_absolute_error(test_data['Close'], forecast)
    rmse = np.sqrt(mean_squared_error(test_data['Close'], forecast))
    print(f"Mean Absolute Error (MAE) for {ticker}: {mae}")
    print(f"Root Mean Squared Error (RMSE) for {ticker}: {rmse}")

    mae_list.append(mae)
    rmse_list.append(rmse)

    # Plot residuals and ACF of residuals
    residuals = arimax_result.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title(f'Residuals of ARIMAX model for {ticker}')
    plt.show()
    plot_acf(residuals, lags=40)
    plt.title(f'Autocorrelation of Residuals for {ticker}')
    plt.show()

# Calculate and display average MAE and RMSE
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)
print(f"\nFinal Average MAE for all tickers: {average_mae}")
print(f"Final Average RMSE for all tickers: {average_rmse}")
