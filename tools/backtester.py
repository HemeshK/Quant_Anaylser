import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class PCABacktester:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    def calculate_residuals(self, returns_df):
        # 1. Standardize the returns
        mu = returns_df.mean()
        std = returns_df.std()
        norm_returns = (returns_df - mu) / std

        # 2. Fit PCA
        self.pca.fit(norm_returns)
        factors = self.pca.transform(norm_returns) 

        # FIXED: Ensure pd is defined here
        residuals = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)

        for stock in returns_df.columns:
            model = LinearRegression()
            model.fit(factors, norm_returns[stock])
            prediction = model.predict(factors)
            residuals[stock] = norm_returns[stock] - prediction

        return residuals

    def compute_signals(self, residuals):
        # Rolling 20-day Z-Score logic
        z_scores = (residuals - residuals.rolling(20).mean()) / residuals.rolling(20).std()
        return z_scores