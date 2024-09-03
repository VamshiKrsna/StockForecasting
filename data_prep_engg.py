import pandas as pd
import numpy as np

def add_features(df):
    """
    Add features to the stock DataFrame.
    """
    # Drop unnecessary columns
    df = df.drop(["Dividends", "Stock Splits"], axis=1, errors='ignore')

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # Extract date features
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    # Set Date as index
    df.set_index('Date', inplace=True)

    # Add rolling mean features
    df['S_3'] = df['Close'].rolling(window=3).mean()
    df['S_9'] = df['Close'].rolling(window=9).mean()
    df['S_18'] = df['Close'].rolling(window=18).mean()

    # Add lag features
    for i in range(1, 4):
        df[f'lag_{i}'] = df['Close'].shift(i)

    # Add rolling window features
    df['Rolling_Mean'] = df['Close'].rolling(window=3).mean()
    df['Rolling_Min'] = df['Close'].rolling(window=3).min()
    df['Rolling_Max'] = df['Close'].rolling(window=3).max()

    # Add Exponential Moving Averages
    df["EMA_12"] = df['Close'].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df['Close'].ewm(span=26, adjust=False).mean()

    # Calculate MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss.replace(0, np.nan)  # Avoid division by zero
    df['RSI'] = 100 - (100 / (1 + RS))

    # Add overall statistics
    df['Overall_Mean'] = df['Close'].mean()
    df['Overall_Min'] = df['Close'].min()
    df['Overall_Max'] = df['Close'].max()

    # Add target variable (next day's Close price)
    df['Target'] = df['Close'].shift(-1)

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    # Process the Tesla stock data
    file_path = "tesla_stock_data.csv"
    df = pd.read_csv(file_path)
    df = add_features(df)
    df.to_csv(file_path, index=False)
    print("Features added for TSLA")
    # print(df.head(5))  # Uncomment to see the first few rows
