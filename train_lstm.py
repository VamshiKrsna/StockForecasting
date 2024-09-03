import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import pickle 

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Assuming 'Date' column exists and has been converted to datetime
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True)
    
    # Use 'Close' price for forecasting
    data = df[['Close']].values

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    seq_length = 60  # Number of time steps to look back
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data into training and testing sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler

# Build and compile LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_log_model(X_train, y_train, X_test, y_test, scaler):
    # Initialize MLFlow
    mlflow.set_experiment("Tesla Stock Forecasting")

    with mlflow.start_run(run_name="LSTM Model Training") as run:
        # Build and train the model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping]
        )
        
        # Evaluate the model
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss}")

        # Log model and parameters to MLFlow
        mlflow.log_param("seq_length", 60)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 50)
        mlflow.log_metric("test_loss", loss)
        
        mlflow.keras.log_model(model, "model")
        
        # Save the scaler for future use
        scaler_path = "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(scaler_path)

if __name__ == "__main__":
    file_path = "tesla_stock_data.csv"
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(file_path)
    train_and_log_model(X_train, y_train, X_test, y_test, scaler)
