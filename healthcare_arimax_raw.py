import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller  # ADF test for stationarity
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # For automated ARIMA parameter tuning
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


# Function to check stationarity using ADF test and apply differencing until stationarity or max differencing
def make_stationary(series, name="", max_diff_order=3):
    differencing_order = 0  # Track the number of differencing steps applied

    while differencing_order <= max_diff_order:
        # Perform the ADF test
        adf_result = adfuller(series)
        print(f"ADF Test Statistic for {name} (Differencing Order: {differencing_order}): {adf_result[0]}")
        print(f"ADF p-value for {name} (Differencing Order: {differencing_order}): {adf_result[1]}")

        # Check if the series is stationary
        if adf_result[1] <= 0.05:
            print(f"The series '{name}' is stationary (p <= 0.05) at differencing order {differencing_order}.")
            return series  # Return the stationary series
        elif differencing_order == max_diff_order:
            print(f"Reached max differencing order for '{name}'. Proceeding with current differencing.")
            return series  # Return the current differenced series even if not stationary
        else:
            # If not stationary, apply further differencing
            print(f"The series '{name}' is NOT stationary (p > 0.05), applying another differencing.")
            series = series.diff().dropna()  # Apply differencing and remove NaN values
            differencing_order += 1  # Increment differencing order


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
    print(f"Raw data for {ticker} saved to '{raw_csv_filename}'")

    # Ensure index is a DateTimeIndex and set frequency
    hist.index = pd.to_datetime(hist.index)
    hist = hist.asfreq('B')  # Set frequency to Business Days ('B')
    hist.ffill(inplace=True)  # Fill missing values

    # Apply stationarity check and differencing to exogenous variables
    hist['Open_stationary'] = make_stationary(hist['Open'], name="Open")
    hist['Low_stationary'] = make_stationary(hist['Low'], name="Low")
    hist['High_stationary'] = make_stationary(hist['High'], name="High")
    hist['Adj_Close_stationary'] = make_stationary(hist['Adj Close'], name="Adj Close")

    # Generate lagged versions of exogenous variables
    hist['Open_stationary_lag1'] = hist['Open_stationary'].shift(1)
    hist['Low_stationary_lag1'] = hist['Low_stationary'].shift(1)
    hist['High_stationary_lag1'] = hist['High_stationary'].shift(1)
    hist['Adj_Close_stationary_lag1'] = hist['Adj_Close_stationary'].shift(1)

    # Drop rows with NaN values created by differencing and lagging
    hist.dropna(inplace=True)

    # Train-test split (80% train, 20% test)
    split_point = int(len(hist) * 0.8)
    train_data = hist[:split_point]
    test_data = hist[split_point:]

    # Define the endogenous variable (raw Close price) and exogenous (lagged stationary variables)
    endog_train = train_data['Close']  # Use the raw Close prices
    exog_train = train_data[['Open_stationary_lag1', 'Low_stationary_lag1', 'High_stationary_lag1', 'Adj_Close_stationary_lag1']]
    endog_test = test_data['Close']  # Use the raw Close prices for testing
    exog_test = test_data[['Open_stationary_lag1', 'Low_stationary_lag1', 'High_stationary_lag1', 'Adj_Close_stationary_lag1']]

    # Auto ARIMA parameter tuning for each ticker
    print(f"Finding optimal ARIMA parameters for {ticker} using auto_arima...")
    auto_arima_model = auto_arima(endog_train, exogenous=exog_train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    best_order = auto_arima_model.order
    print(f"Optimal ARIMA order for {ticker}: {best_order}")

    # Fit ARIMAX model with tuned parameters
    print(f"Fitting ARIMAX model for {ticker} with order {best_order}...")
    try:
        model = ARIMA(endog_train, order=best_order, exog=exog_train)
        arimax_result = model.fit()  # Fit without maxiter to rely on default optimizer settings
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as e:
        print(f"Error fitting ARIMAX model for {ticker} with order {best_order}: {e}. Trying a simpler model order.")
        try:
            # If error occurs, fallback to a simpler ARIMA order, e.g., (1,1,1)
            simpler_order = (1, 1, 1)
            model = ARIMA(endog_train, order=simpler_order, exog=exog_train)
            arimax_result = model.fit()  # Use the simpler model without maxiter
            print(f"Simpler ARIMA model with order {simpler_order} fitted for {ticker}.")
        except Exception as final_e:
            print(f"Failed to fit ARIMAX model for {ticker} even with simpler order: {final_e}")
            continue  # Skip this ticker if all fitting attempts fail

    # Forecast using the ARIMAX model
    forecast = arimax_result.forecast(steps=len(endog_test), exog=exog_test)

    # Plot actual vs forecasted values (in original 'Close' scale)
    plt.figure(figsize=(10, 6))
    plt.plot(endog_test.index, test_data['Close'], label='Actual Close Prices', color='blue')
    plt.plot(endog_test.index, forecast, label='Forecasted Close Prices', color='red')
    plt.title(f'ARIMAX Forecast vs Actual for {ticker}')
    plt.legend()
    plt.show()

    # Calculate MAE and RMSE directly since no inversion is required
    mae = mean_absolute_error(test_data['Close'], forecast)
    rmse = np.sqrt(mean_squared_error(test_data['Close'], forecast))
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

print("Data processing, ADF testing, differencing, lagging, ARIMAX modeling with auto-tuning, and residual analysis completed.")

# Calculate the average MAE and RMSE
average_mae = np.mean(mae_list)
average_rmse = np.mean(rmse_list)

# Display the final average MAE and RMSE
print(f"\nFinal Average MAE for all tickers: {average_mae}")
print(f"Final Average RMSE for all tickers: {average_rmse}")
