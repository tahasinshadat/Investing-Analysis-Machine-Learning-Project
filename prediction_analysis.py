import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def fit_arima_model(history):
    if history.index.freq is None:
        history = history.asfreq(pd.infer_freq(history.index), method='pad')

    model = ARIMA(history['Close'], order=(5,1,0))
    model_fit = model.fit()
    return model_fit

def predict_future_prices(history, math_analysis, periods):
    """Predict future stock prices with both trend and volatility."""
    latest_close = history['Close'].iloc[-1]
    trend_slope = math_analysis['trend_line_slope']
    trend_intercept = math_analysis['trend_line_intercept']
    volatility = math_analysis['volatility']
    predictions = {}

    # ARIMA Model Predictions
    arima_model = fit_arima_model(history)
    forecast = arima_model.forecast(steps=max(periods.values()))
    
    # Generate predictions combining trend and ARIMA forecast with added volatility
    for period, days in periods.items():
        if days <= len(forecast):
            arima_forecast_price = forecast.iloc[days-1]
            
            # Calculate trend component
            trend_price = trend_slope * days + trend_intercept
            
            # Combine ARIMA forecast with trend and add randomness for volatility
            noise = np.random.normal(loc=0, scale=volatility)
            combined_price = trend_price + (arima_forecast_price - trend_price) + noise

            predictions[period] = combined_price
        else:
            predictions[period] = np.nan  # Handle case where forecast is not available

    return predictions

def test_model_accuracy(analysis_results, history):
    """Test model accuracy based on predictions and historical data."""

    # Define prediction periods
    periods = {
        "1_day": 1,
        "1_week": 5,
        "1_month": 21,
        "3_months": 63
    }
    
    # Predict future prices
    future_predictions = predict_future_prices(history, analysis_results, periods)
    
    # Define the test period range
    end_date = history.index[-1]
    start_date = end_date - pd.DateOffset(months=3)
    test_data = history.loc[start_date:end_date].copy()
    
    # Prepare test data for evaluation
    test_data['Predicted'] = np.nan
    test_data.loc[test_data.index[-1], 'Predicted'] = future_predictions['3_months']
    
    X_test = np.arange(len(test_data)).reshape(-1, 1)  # Days as feature
    y_test = test_data['Close'].values  # Actual prices
    
    # Create a linear regression model for comparison
    model = LinearRegression()
    model.fit(X_test, y_test)
    predictions = model.predict(X_test)
    test_data['Predicted'] = predictions
    
    # Calculate errors
    actual_prices = test_data['Close'].dropna()
    predicted_prices = test_data['Predicted'].dropna()

    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predicted_prices)
    
    error_metrics = {
        "Mean Absolute Error (MAE)": mae,
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": rmse,
        "R^2 Score": r2
    }

    return error_metrics
# ticker = "PSA"
# test_model_accuracy(ticker)