import yfinance as yf
import os

def fetch_tesla_stock_data():
    """
    Fetch Tesla stock data and store it in the specified directory.
    """
    # Define the stock ticker symbol for Tesla
    ticker_symbol = "TSLA"

    # Define the directory and file path
    
    file_path = "tesla_stock_data.csv"

    # Fetch the Tesla stock data
    print("Fetching data for Tesla...")
    tesla_data = yf.download(ticker_symbol, start="2015-01-01", end="2024-12-31")
    
    # Save the data to a CSV file
    tesla_data.to_csv(file_path)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    fetch_tesla_stock_data()
