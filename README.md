# DSE4212 Project: Stock Price Prediction

This repository contains code and data for the DSE4212 Group Project by Group 14, which focuses on predicting next-day stock prices using various machine learning models. The project compares traditional time-series models with advanced neural network architectures to analyze performance across technology and healthcare sectors.

## Project Overview

The goal of this project is to evaluate the predictive accuracy of different models, including:
- **ARIMA** (traditional time-series model)
- **CNN** (Convolutional Neural Network)
- **LSTM** (Long Short-Term Memory)
- **CNN-LSTM Hybrid** (combining CNN and LSTM)

The study is based on the hypothesis that neural networks, especially CNN and LSTM, can outperform ARIMA due to their ability to capture complex patterns in financial data. However, the project also considers the computational trade-offs associated with these models.

## Data Description

We selected six major U.S. stocks from each of the technology and healthcare sectors for their market influence and contrasting volatility patterns:
- **Technology**: Apple (AAPL), Microsoft (MSFT), Google (GOOG), Amazon (AMZN), Meta (META), NVIDIA (NVDA)
- **Healthcare**: Eli Lilly (LLY), AbbVie (ABBV), Johnson & Johnson (JNJ), Merck (MRK), Thermo Fisher Scientific (TMO), UnitedHealth Group (UNH)

Data was sourced from Yahoo Finance via the yfinance library, covering a 10-year period from September 1, 2014, to September 1, 2024. Each stock dataset includes daily values for Open, High, Low, Volume, and Adjusted Close prices, with Close prices used as the target variable for prediction.

## Data Preprocessing

The data preprocessing steps involved:
1. **Feature Engineering**: Creation of indicators like moving averages, Relative Strength Index (RSI), Average True Range (ATR), and lagged variables.
2. **Normalization**: Using MinMax scaling to normalize feature values.
3. **Train-Test Split**: An 80-20 split was used to train and evaluate the models.

## Model Descriptions

1. **ARIMA**  
   A traditional time-series model, ARIMA served as the benchmark by capturing temporal dependencies. The pmdarima library was used for automatic parameter tuning.

2. **CNN**  
   A CNN model was used to detect complex patterns over small time windows in stock price data. The CNN architecture includes:
   - Convolutional layer (64 filters, kernel size of 2)
   - MaxPooling layer
   - Dropout and Dense layers for feature extraction

3. **LSTM**  
   LSTM is suitable for capturing long-term dependencies in sequential data. The model was structured with:
   - Two LSTM layers
   - Dropout layers to mitigate overfitting

4. **CNN-LSTM Hybrid**  
   The hybrid model combines CNN's ability to extract local features with LSTM's capacity to model long-term dependencies, aiming to improve prediction accuracy for complex time-series data.

## Methodology

Model performance was evaluated based on:
- **Root Mean Square Error (RMSE)**
- **Mean Absolute Error (MAE)**

The data pipeline includes scripts for preprocessing, feature engineering, and training the models.

## Folder Structure

```
.
├── data/
├── src/
│   ├── CNN_raw.ipynb
│   ├── LSTM_raw.ipynb
│   ├── CNN-LSTM_raw.ipynb
│   └── arima_raw.ipynb
├── annex/
└── README.md
```

- `data` - Contains raw and processed stock data for both technology and healthcare sectors.
- `src`: Code files for each model type, including preprocessing steps.
  - `CNN_raw.ipynb`: CNN model using raw variables.
  - `LSTM_raw.ipynb`: LSTM model with raw variables.
  - `CNN-LSTM_raw.ipynb`: Hybrid CNN-LSTM model with raw variables.
  - `arimax_raw.ipynb`: ARIMA model with exogenous variables.
- `annex`: Contains full model variations with additional feature engineering.

## Results

The results demonstrated that:
- Neural network models (CNN, LSTM, CNN-LSTM) consistently outperformed ARIMA.
- The CNN model achieved the highest accuracy, followed closely by LSTM.
- The CNN-LSTM hybrid model did not perform as expected, possibly due to overfitting from increased complexity.

## Key Insights

- **Model Performance**: Simple neural architectures (CNN and LSTM) provided better accuracy compared to the more complex CNN-LSTM.
- **Sector Variability**: The models faced more challenges in predicting healthcare stocks due to the unique behaviors of individual stocks like LLY, which experienced extreme price movements.
- **Feature Engineering**: Adding technical indicators (e.g., SMA, RSI) did not significantly improve performance, suggesting that simpler raw metrics (Open, High, Low, Volume, Close) are effective for this task.

## Future Work

Potential areas for improvement:
- **Hyperparameter Tuning**: Using systematic hyperparameter tuning for neural networks could enhance performance.
- **Qualitative Factors**: Incorporating sentiment analysis could improve predictions, especially for volatile technology stocks.
- **Additional Sectors**: Expanding to other sectors like energy and finance to assess model robustness across different industries.
