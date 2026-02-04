"""
Data loading module for FX rates from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import os


class FXDataLoader:
    """
    Loader for FX rate data from Yahoo Finance.
    
    Yahoo Finance FX tickers format:
    - EURUSD=X (EUR/USD)
    - GBPUSD=X (GBP/USD)
    - USDJPY=X (USD/JPY)
    - etc.
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_fx_data(
        self,
        pairs: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        freq: str = "1d",
        column: str = "Close"
    ) -> pd.DataFrame:
        """
        Fetch FX rate data from Yahoo Finance.
        
        Args:
            pairs: List of FX pair tickers (e.g., ["EURUSD=X", "GBPUSD=X"])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (default: today)
            freq: Data frequency (1d, 1wk, 1mo)
            column: Which price column to use (Open, High, Low, Close, Adj Close)
        
        Returns:
            DataFrame with FX rates, columns are FX pairs
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        data_dict = {}
        
        for pair in pairs:
            print(f"Fetching {pair}...")
            try:
                ticker = yf.Ticker(pair)
                hist = ticker.history(start=start_date, end=end_date, interval=freq)
                
                if not hist.empty:
                    # Clean up pair name for column
                    clean_name = pair.replace("=X", "")
                    data_dict[clean_name] = hist[column]
                else:
                    print(f"Warning: No data found for {pair}")
                    
            except Exception as e:
                print(f"Error fetching {pair}: {e}")
        
        df = pd.DataFrame(data_dict)
        df.dropna(how='all', inplace=True)
        
        return df
    
    def load_or_fetch(
        self,
        pairs: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        freq: str = "1d",
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Load from cache if available, otherwise fetch and cache.
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"fx_data_{'_'.join(pairs)}_{start_date}_{end_date or 'today'}_{freq}.csv"
        )
        
        if cache and os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        df = self.fetch_fx_data(pairs, start_date, end_date, freq)
        
        if cache:
            df.to_csv(cache_file)
            print(f"Saved to cache: {cache_file}")
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Returns:
            train_df, val_df, test_df
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_common_fx_pairs(self) -> List[str]:
        """Get list of commonly traded FX pairs."""
        return [
            "EURUSD=X",   # Euro/US Dollar
            "GBPUSD=X",   # British Pound/US Dollar
            "USDJPY=X",   # US Dollar/Japanese Yen
            "USDCHF=X",   # US Dollar/Swiss Franc
            "AUDUSD=X",   # Australian Dollar/US Dollar
            "USDCAD=X",   # US Dollar/Canadian Dollar
            "NZDUSD=X",   # New Zealand Dollar/US Dollar
            "EURGBP=X",   # Euro/British Pound
            "EURJPY=X",   # Euro/Japanese Yen
            "GBPJPY=X",   # British Pound/Japanese Yen
        ]


if __name__ == "__main__":
    # Test the data loader
    loader = FXDataLoader()
    
    pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    df = loader.fetch_fx_data(pairs, "2022-01-01", "2023-12-31")
    
    print("\nData preview:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")