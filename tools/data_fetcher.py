import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def get_market_data(self, tickers, start="2024-01-01", end="2026-02-01"):
        """Downloads prices. auto_adjust=False ensures 'Adj Close' is present."""
        print(f"--- Downloading data for {len(tickers)} assets ---")
        
        # We add auto_adjust=False to keep the 'Adj Close' column
        data = yf.download(tickers, start=start, end=end, auto_adjust=False)
        
        # Because yfinance returns a MultiIndex (Price, Ticker), we extract just Adj Close
        if 'Adj Close' in data.columns:
            adj_close_data = data['Adj Close']
        else:
            # Fallback for newer versions where they might use 'Close' as adjusted
            print("Warning: 'Adj Close' not found, using 'Close' instead.")
            adj_close_data = data['Close']
            
        # Drop assets with no data and fill gaps
        adj_close_data = adj_close_data.dropna(axis=1, how='all')
        adj_close_data = adj_close_data.ffill().dropna()
        
        return adj_close_data

    def get_sp500_tickers(self):
        """Quick helper to get S&P 500 tickers via Wikipedia."""
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].tolist()
    