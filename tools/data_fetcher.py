import yfinance as yf
import pandas as pd

class DataFetcher:
    @staticmethod
    def get_data(spec: dict):
        ticker = spec.get('ticker', 'SPY')
        try:
            df = yf.download(ticker, period="1y", interval="1d")
            if df.empty:
                return {"error": f"No data found for {ticker}"}
            return df
        except Exception as e:
            return {"error": str(e)}