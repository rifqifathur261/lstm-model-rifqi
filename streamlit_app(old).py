import streamlit as st
import yfinance as yf
from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.express as go
from tensorflow.keras.models import load_model
import tf_keras as k3
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import plotly.express as px

# Memuat model yang telah dilatih
pipeline = joblib.load('stock_price_pipeline_updated.pkl')
pipeline2 = joblib.load('trained_lstm_model_rifqi2.pkl')

# Fungsi untuk menghitung fitur tambahan
def calculate_features(data):
    data['MA_3'] = data['Close'].rolling(window=3).mean().shift(1)
    data['MA_5'] = data['Close'].rolling(window=5).mean().shift(1)
    data['MA_10'] = data['Close'].rolling(window=10).mean().shift(1)
    data['Return'] = data['Close'].pct_change().shift(1)
    data['Volatility'] = data['Return'].rolling(window=10).std().shift(1)
    data = data.fillna(0)
    return data

# Judul aplikasi
st.title('Prediksi Harga Saham')

# Sidebar for data download
st.sidebar.header("Data Download")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., ANTM.JK):", "BBCA.JK")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

# Download stock price data
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Data preprocessing
close_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# # Input ticker saham
# st.header('Masukan Kode Emiten')
# ticker = st.text_input('Ticker', value='AAPL')

# Tombol untuk mengambil data dan melakukan prediksi
# if st.button('Predict'):
#     # Mendapatkan data saham 3 bulan terakhir
#     end_date = datetime.today().date()
#     start_date = end_date - timedelta(days=360)
#     stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # if not stock_data.empty:
    #     # Menghitung fitur tambahan untuk seluruh data
    #     stock_data = calculate_features(stock_data)
        
    #     # Mengambil data terbaru setelah menghitung fitur tambahan
    #     latest_data = stock_data.iloc[-1]
    #     Open = latest_data['Open']
    #     High = latest_data['High']
    #     Low = latest_data['Low']
    #     Close = latest_data['Close']
    #     Volume = latest_data['Volume']

    #     # Membuat DataFrame untuk prediksi
    #     input_data = pd.DataFrame({
    #         'Open': [Open],
    #         'High': [High],
    #         'Low': [Low],
    #         'Close': [Close],
    #         'Volume': [Volume],
    #         'MA_3': [latest_data['MA_3']],
    #         'MA_5': [latest_data['MA_5']],
    #         'MA_10': [latest_data['MA_10']],
    #         'Return': [latest_data['Return']],
    #         'Volatility': [latest_data['Volatility']]
    #     })
        
    #     # Melakukan prediksi menggunakan pipeline
    #     prediction = pipeline.predict(input_data)

        # Menampilkan grafik garis harga penutupan 3 bulan terakhir
        # fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'1 Year Close Prices for {ticker}')
        # st.plotly_chart(fig)

        # Menampilkan harga penutupan terbaru dalam format Rupiah
        # st.subheader(f'Latest Close Price for {ticker}: Rp{Close:.2f}')

        # # Menampilkan hasil prediksi dalam format Rupiah
        # st.subheader(f'Predicted Close Price for {ticker}: Rp{prediction[0]:.2f}')

    # else:
    #     st.error(f'No data found for ticker: {ticker}')

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        lag_values = data[i:(i + n_steps), 0]
        X.append(np.concatenate([lag_values, [data[i + n_steps, 0]]]))
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

def visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                                y=y_test_orig.flatten(),
                                mode='lines',
                                name="Actual Stock Prices",
                                line=dict(color='blue')))

        fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                                y=y_pred.flatten(),
                                mode='lines',
                                name="Predicted Stock Prices",
                                line=dict(color='red')))

        fig.update_layout(title="Stock Price Prediction",
                        xaxis_title="Date",
                        yaxis_title="Stock Price (IDR)",
                        template='plotly_dark')

        st.plotly_chart(fig)

# def load_model(filename='trained_lstm_model_rifqi2.pkl'):
#     with open(filename, 'rb') as f:
#         data = pickle.load(f)
#     return data['model'], data['scaler']

if st.button('Predict'):
 # Data preparation
    # n_steps = 120
    # X, y = prepare_data(scaled_data, n_steps)

    # # Splitting into train and test sets
    # train_size = int(len(X) * 0.8)
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]

    # # Reshape data for LSTM and GRU models
    # X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # X_train_gru = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    # X_test_gru = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # # Sidebar for model selection
    # st.sidebar.header("Select Model")
    # model_type = st.sidebar.selectbox("Select Model Type:", ["LSTM", "GRU"])

    #  Mendapatkan data saham 3 bulan terakhir
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=360)
    stock_data = yf.download('ANTM.JK', start=start_date, end=end_date)
    
    if not stock_data.empty:
        # Menghitung fitur tambahan untuk seluruh data
        stock_data = calculate_features(stock_data)
        
        # Mengambil data terbaru setelah menghitung fitur tambahan
        latest_data = stock_data.iloc[-1]
        Open = latest_data['Open']
        High = latest_data['High']
        Low = latest_data['Low']
        Close = latest_data['Close']
        Volume = latest_data['Volume']
        
       

        # Membuat DataFrame untuk prediksi
        input_data = pd.DataFrame({
            'Open': [Open],
            'High': [High],
            'Low': [Low],
            'Close': [Close],
            'Volume': [Volume],
            'MA_3': [latest_data['MA_3']],
            'MA_5': [latest_data['MA_5']],
            'MA_10': [latest_data['MA_10']],
            'Return': [latest_data['Return']],
            'Volatility': [latest_data['Volatility']]
        })
        
    st.write("Ho ", latest_data)
    # Load saved models
    # final_model = pipeline.predict(input_data)
    # final_model = k3.models.load_model("trained_lstm_model_rifqi2.pkl")
   
    

    # Model evaluation
    # y_pred = final_model.predict(X_test)
    y_pred = pipeline.predict(input_data)
    # y_pred = scaler.inverse_transform(y_pred)
    # y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate metrics
    # mse = mean_squared_error(y_test_orig, y_pred)
    # rmse = math.sqrt(mse)
    # mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

    # Display results
    # st.header(f"Results for {model_type} Model")
    # st.write("Mean Squared Error (MSE):", mse)
    # st.write("Root Mean Squared Error (RMSE):", rmse)
    # st.write("Mean Absolute Percentage Error (MAPE):", mape)

    # Visualize predictions
    st.header("Visualize Predictions")
    # fig = go.Figure()
    fig = px.line(stock_data, x=stock_data.index, y=stock_data.columns, title=f'1 Year Close Prices for {"ANTM.JK"}')
    st.plotly_chart(fig)
    # visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred)