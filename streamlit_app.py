import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import load_model
from datetime import datetime

# mengambil dataset dari yahoo finance
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1), data

# Function to create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# membuat judul aplikasi
st.title('Prediksi Harga Saham')

# masukan dari user berupa kode emiten dan tanggal
ticker = st.text_input('Masukan kode emiten', 'ANTM.JK')
start_date = st.date_input('Tanggal Awal', datetime.today().replace(year=datetime.today().year - 5))
end_date = st.date_input('Tanggal Akhir', datetime.today())

if st.button('Prediksi'):
    # mendapatkan data baru
    new_data, data = get_stock_data(ticker, start_date, end_date)
    
    # penskalaan data menjadi 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_new_data = scaler.fit_transform(new_data)
    
    # membagi data menjadi 80% data training dan 20% data testing
    train_size = int(len(scaled_new_data) * 0.8)
    test_size = len(scaled_new_data) - train_size
    train_data = scaled_new_data[:train_size]
    test_data = scaled_new_data[train_size - 60:]  # 60-day lookback
    
    # Create sequences for training and testing
    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)
    
    # Reshape for LSTM input (samples, time_steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Load the model
    model = load_model('lstm_model.h5')
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # # Calculate RMSE
    # rmse = np.sqrt(np.mean((predictions - y_test_actual)**2))
    # st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Calculate R-squared
    r2 = r2_score(y_test_actual, predictions)
    st.write(f"R-squared: {r2:.2f}")
    
    # Get the corresponding dates for the test data
    test_dates = data.index[-test_size:]
    
    # Plot results
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data['Close'], color='blue', label='Actual Price')
    plt.plot(test_dates, predictions[-test_size:], color='red', label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Use the last 60 days to predict the next day
    last_sequence = scaled_new_data[-60:]
    last_sequence = np.reshape(last_sequence, (1, 60, 1))
    future_price_scaled = model.predict(last_sequence)
    future_price = scaler.inverse_transform(future_price_scaled)
    st.write(f"Prediksi harga di hari selanjutnya: {future_price[0][0]:.2f}")
