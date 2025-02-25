import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# Define stock symbol and date range (e.g., Apple from 2020 to 2025)
stock_symbol = 'ANTM.JK'
start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

data = yf.download(stock_symbol, start=start_date, end=end_date)
print(data.head())

close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Split 80% for training, 20% for testing
train_size = int(len(scaled_prices) * 0.8)
test_size = len(scaled_prices) - train_size
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size - 60:]  # 60-day lookback

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Reshape for LSTM input (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, 
                    batch_size=32, 
                    epochs=50, 
                    validation_data=(X_test, y_test))

# Save the model
model.save('lstm_model.h5')

# Load the model (for demonstration purposes)
# model = load_model('/Users/nasri/Workspace/Personal/LSTM/lstm_model.h5')

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(np.mean((predictions - y_test_actual)**2))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Get the corresponding dates for the test data (last 3 months)
test_dates = data.index[-test_size:]

plt.figure(figsize=(12,6))
plt.plot(data.index, data['Close'], color='blue', label='Actual Price')
plt.plot(test_dates, predictions[-test_size:], color='red', label='Predicted Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Use the last 60 days to predict the next day
last_sequence = scaled_prices[-60:]
last_sequence = np.reshape(last_sequence, (1, 60, 1))
future_price_scaled = model.predict(last_sequence)
future_price = scaler.inverse_transform(future_price_scaled)
print(f"Next-day predicted price: ${future_price[0][0]:.2f}")
