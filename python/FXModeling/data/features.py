"""
Feature engineering for FX time series data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureEngineer:
    """
    Feature engineering for FX rate time series.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scalers = {}
    
    def create_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create simple returns."""
        returns_df = df.pct_change().add_suffix('_return')
        return pd.concat([df, returns_df], axis=1)
    
    def create_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create log returns."""
        log_returns = np.log(df / df.shift(1)).add_suffix('_log_return')
        return pd.concat([df, log_returns], axis=1)
    
    def create_volatility(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """Create rolling volatility features."""
        result = df.copy()
        for col in df.columns:
            returns = df[col].pct_change()
            for window in windows:
                result[f'{col}_vol_{window}'] = returns.rolling(window).std()
        return result
    
    def create_sma(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """Create simple moving average features."""
        result = df.copy()
        for col in df.columns:
            for window in windows:
                result[f'{col}_sma_{window}'] = df[col].rolling(window).mean()
        return result
    
    def create_ema(self, df: pd.DataFrame, windows: List[int] = [12, 26]) -> pd.DataFrame:
        """Create exponential moving average features."""
        result = df.copy()
        for col in df.columns:
            for window in windows:
                result[f'{col}_ema_{window}'] = df[col].ewm(span=window, adjust=False).mean()
        return result
    
    def create_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Create Relative Strength Index."""
        result = df.copy()
        for col in df.columns:
            delta = df[col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            result[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        return result
    
    def create_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Create MACD (Moving Average Convergence Divergence)."""
        result = df.copy()
        for col in df.columns:
            ema_fast = df[col].ewm(span=fast, adjust=False).mean()
            ema_slow = df[col].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal, adjust=False).mean()
            result[f'{col}_macd'] = macd
            result[f'{col}_macd_signal'] = macd_signal
            result[f'{col}_macd_hist'] = macd - macd_signal
        return result
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """Create lagged price features."""
        result = df.copy()
        for col in df.columns:
            for lag in lags:
                result[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return result
    
    def create_all_features(self, df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create all features based on configuration.
        """
        cfg = config or self.config
        result = df.copy()
        
        # Returns
        if cfg.get('returns', True):
            returns_df = self.create_returns(df)
            result = pd.concat([result, returns_df.filter(like='_return')], axis=1)
        
        # Log returns
        if cfg.get('log_returns', True):
            log_ret_df = self.create_log_returns(df)
            result = pd.concat([result, log_ret_df.filter(like='_log_return')], axis=1)
        
        # Volatility
        if 'volatility_windows' in cfg:
            vol_df = self.create_volatility(df, cfg['volatility_windows'])
            result = pd.concat([result, vol_df.filter(like='_vol_')], axis=1)
        
        # SMA
        if 'sma_windows' in cfg:
            sma_df = self.create_sma(df, cfg['sma_windows'])
            result = pd.concat([result, sma_df.filter(like='_sma_')], axis=1)
        
        # EMA
        if 'ema_windows' in cfg:
            ema_df = self.create_ema(df, cfg['ema_windows'])
            result = pd.concat([result, ema_df.filter(like='_ema_')], axis=1)
        
        # RSI
        if 'rsi_period' in cfg:
            rsi_df = self.create_rsi(df, cfg['rsi_period'])
            result = pd.concat([result, rsi_df.filter(like='_rsi')], axis=1)
        
        # MACD
        if all(k in cfg for k in ['macd_fast', 'macd_slow', 'macd_signal']):
            macd_df = self.create_macd(df, cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal'])
            result = pd.concat([result, macd_df.filter(like='_macd')], axis=1)
        
        # Lag features
        if 'lags' in cfg:
            lag_df = self.create_lag_features(df, cfg['lags'])
            result = pd.concat([result, lag_df.filter(like='_lag_')], axis=1)
        
        return result
    
    def scale_features(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        method: str = "standard"
    ) -> tuple:
        """
        Scale features using fitted scaler on training data.
        """
        scaler_class = StandardScaler if method == "standard" else MinMaxScaler
        
        # Fit on training data
        scaler = scaler_class()
        train_scaled = pd.DataFrame(
            scaler.fit_transform(train_df),
            columns=train_df.columns,
            index=train_df.index
        )
        self.scalers['features'] = scaler
        
        # Transform validation and test
        val_scaled = None
        test_scaled = None
        
        if val_df is not None:
            val_scaled = pd.DataFrame(
                scaler.transform(val_df),
                columns=val_df.columns,
                index=val_df.index
            )
        
        if test_df is not None:
            test_scaled = pd.DataFrame(
                scaler.transform(test_df),
                columns=test_df.columns,
                index=test_df.index
            )
        
        return train_scaled, val_scaled, test_scaled
    
    def create_sequences(
        self,
        data: np.ndarray,
        target_idx: int,
        seq_length: int,
        horizon: int = 1
    ) -> tuple:
        """
        Create sequences for time series forecasting.
        
        Args:
            data: Input data array (n_samples, n_features)
            target_idx: Index of target column
            seq_length: Length of input sequence
            horizon: Prediction horizon
        
        Returns:
            X, y arrays for supervised learning
        """
        X, y = [], []
        for i in range(len(data) - seq_length - horizon + 1):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length + horizon - 1, target_idx])
        
        return np.array(X), np.array(y)


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import FXDataLoader
    
    loader = FXDataLoader()
    df = loader.fetch_fx_data(["EURUSD=X"], "2022-01-01", "2023-12-31")
    
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(df, {
        'returns': True,
        'log_returns': True,
        'volatility_windows': [5, 20],
        'sma_windows': [10, 20],
        'rsi_period': 14
    })
    
    print(f"Original shape: {df.shape}")
    print(f"Features shape: {features_df.shape}")
    print(f"\nFeature columns:\n{features_df.columns.tolist()}")