import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller  # ADF test for stationarity
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For ACF and PACF plotting
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

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

# Function to check stationarity using ADF test
def check_stationarity(series, name=""):
    adf_result = adfuller(series)
    print(f"ADF Test Statistic for {name}: {adf_result[0]}")
    print(f"ADF p-value for {name}: {adf_result[1]}")
    if adf_result[1] <= 0.05:
        print(f"The series '{name}' is stationary (p <= 0.05).")
    else:
        print(f"The series '{name}' is NOT stationary (p > 0.05).")

# Initialize lists to store RMSE and MAE for each ticker
mae_list = []
rmse_list = []

# Loop through each ticker, download the data, process it, and save to CSV
for ticker in tickers:
    print(f"\nProcessing data for {ticker}...")

    # Download historical stock data using yf.download()
    hist = yf.download(ticker, start=start_date, end=end_date)

    # Save the raw data to a CSV file before any processing
    raw_csv_filename = os.path.join(data_folder, f"{ticker}_raw_data.csv")
    hist.to_csv(raw_csv_filename)
    print(f"Raw data for {ticker} saved to '{raw_csv_filename}'")

    # Ensure the index is a DateTimeIndex and set the frequency
    hist.index = pd.to_datetime(hist.index)  # Ensure the index is DateTimeIndex
    hist = hist.asfreq('B')  # Set frequency to Business Days ('B')

    # Fill any missing values caused by business day frequency adjustment
    hist.ffill(inplace=True)  # Forward fill to handle any missing dates

    # Check stationarity of the 'Close' prices
    print(f"\nChecking stationarity for {ticker} 'Close' prices:")
    check_stationarity(hist['Close'], name="Close")

    # Apply differencing to make the series stationary
    hist['Close_diff'] = hist['Close'].diff().dropna()  # First-order differencing

    # Check stationarity of the differenced 'Close' prices
    print(f"\nChecking stationarity for differenced {ticker} 'Close' prices:")
    check_stationarity(hist['Close_diff'].dropna(), name="Close_diff")

    # Plot ACF and PACF for the differenced series
    print(f"\nPlotting ACF and PACF for differenced {ticker} 'Close' prices...")
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

    # Continue with model fitting using ARIMAX if p, d, q are determined (defaulting to 1, 1, 1 for now)
    # You can change p, d, q based on the visual inspection of ACF and PACF plots.
    p, d, q = 1, 1, 1  # Defaulting to 1,1,1 for simplicity (adjust based on your ACF/PACF inspection)

    # Train-test split (80% train, 20% test)
    split_point = int(len(hist) * 0.8)
    train_data = hist[:split_point]
    test_data = hist[split_point:]

    # Define endogenous (Close price) and exogenous (Open, Low, High)
    endog_train = train_data['Close']
    exog_train = train_data[['Open', 'Low', 'High']]
    endog_test = test_data['Close']
    exog_test = test_data[['Open', 'Low', 'High']]

    # Fit ARIMAX model with exogenous variables (p, d, q determined via ACF/PACF)
    print(f"Fitting ARIMAX model for {ticker} with order ({p}, {d}, {q}) and exogenous variables...")
    model = ARIMA(endog_train, order=(p, d, q), exog=exog_train)
    arimax_result = model.fit()

    # Forecast the values for the test set
    forecast = arimax_result.forecast(steps=len(endog_test), exog=exog_test)

    # Plot actual vs forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(endog_test.index, endog_test, label='Actual Close Prices', color='blue')
    plt.plot(endog_test.index, forecast, label='Forecasted Close Prices', color='red')
    plt.title(f'ARIMAX Forecast vs Actual for {ticker}')
    plt.legend()
    plt.show()

    # Evaluate the model performance
    mae = mean_absolute_error(endog_test, forecast)
    rmse = np.sqrt(mean_squared_error(endog_test, forecast))
    print(f"Mean Absolute Error (MAE) for {ticker}: {mae}")
    print(f"Root Mean Squared Error (RMSE) for {ticker}: {rmse}")

    # Store the MAE and RMSE in the respective lists
    mae_list.append(mae)
    rmse_list.append(rmse)

    # Plot the residuals to check the fit
    residuals = arimax_result.resid
    plt.figure(figsize=(10, 6))
    plt.plot(residuals)
    plt.title(f'Residuals of ARIMAX model for {ticker}')
    plt.show()

    # Plot the autocorrelation function (ACF) of residuals
    print(f"Plotting ACF of residuals for {ticker}...")
    plot_acf(residuals, lags=40)
    plt.title(f'Autocorrelation of Residuals for {ticker}')
    plt.show()

print("Data processing, ADF testing, ACF/PACF plotting, ARIMAX modeling, and residual analysis completed.")

# Calculate the average MAE and RMSE
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)

# Display the final average MAE and RMSE
print(f"\nFinal Average MAE for all tickers: {average_mae}")
print(f"Final Average RMSE for all tickers: {average_rmse}")