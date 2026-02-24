import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class PCABacktester:
    def __init__(self, spec=None):
        # Extract parameters from the agent's JSON spec, with defaults
        self.spec = spec or {}
        params = self.spec.get("strategy", {}).get("parameters", {})
        
        # Dynamically set PCA factors and rolling window
        self.n_components = params.get("num_factors") or params.get("pca_factors") or 5
        self.window = params.get("window") or 20 
        
        self.pca = PCA(n_components=self.n_components)

    def calculate_residuals(self, returns_df):
        """Decomposes returns based on the dynamic n_components."""
        mu = returns_df.mean()
        std = returns_df.std()
        norm_returns = (returns_df - mu) / std

        self.pca.fit(norm_returns)
        factors = self.pca.transform(norm_returns) 

        residuals = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)

        for stock in returns_df.columns:
            model = LinearRegression()
            model.fit(factors, norm_returns[stock])
            prediction = model.predict(factors)
            residuals[stock] = norm_returns[stock] - prediction

        return residuals

    def compute_signals(self, residuals):
        """Calculates Z-Scores using the dynamic window from the spec."""
        # Uses the window (e.g., 20) provided by the Strategist
        z_scores = (residuals - residuals.rolling(self.window).mean()) / residuals.rolling(self.window).std()
        return z_scores