import yfinance as yf
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_data(tickers, start_date, end_date, interval='1d'):
    """
    Download historical data from Yahoo Finance.
    
    Args:
        tickers (list or str): List of tickers or single ticker string (e.g. ['SPY', 'EURUSD=X']).
        start_date (str): Start date in 'YYYY-MM-DD'.
        end_date (str): End date in 'YYYY-MM-DD'.
        interval (str): Data interval (default '1d').
        
    Returns:
        pd.DataFrame: Adjusted Close prices.
    """
    logger.info(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False)
        logger.info(f"Downloaded data columns: {data.columns}")
        logger.info(f"Downloaded data head: \n{data.head()}")
        
        # Handle cases where multiple tickers return a MultiIndex
        if isinstance(tickers, list) and len(tickers) > 1:
            if 'Adj Close' in data.columns:
                 adj_close = data['Adj Close']
            elif 'Close' in data.columns:
                 adj_close = data['Close']
            else:
                 # Try accessing top level if columns are just tickers (rare)
                 adj_close = data

            # yfinance structure varies by version; safe check
            if 'Adj Close' in data:
                adj_close = data['Adj Close']
            elif 'Close' in data:
                 # Forex sometimes doesn't have Adj Close distinct from Close
                adj_close = data['Close']
            else:
                 raise ValueError("Could not find Close/Adj Close in downloaded data")
                 
        return adj_close
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return pd.DataFrame()

def calculate_log_returns(prices):
    """
    Calculate log returns from prices.
    r_t = ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1)).dropna()

def create_sequences(data, seq_length, target_col=None):
    """
    Create sequences for Deep Learning models.
    X: (N, seq_length, num_features)
    y: (N,) or (N, 1) depending on target
    
    Args:
        data (np.ndarray or pd.DataFrame): Input data. 
        seq_length (int): Length of the input sequence.
        target_col (int, optional): Index of the target column if data is multivariate. 
                                    If None, assumes univariate (target is same as input shifted).
    
    Returns:
        np.array, np.array: X, y
    """
    xs, ys = [], []
    
    # Ensure data is numpy array
    if isinstance(data, pd.DataFrame):
        data_values = data.values
    else:
        data_values = data
        
    for i in range(len(data_values) - seq_length):
        x = data_values[i : i + seq_length]
        # For simplicity, predict the NEXT step's return (or volatility proxy)
        # If strictly predicting next return:
        if target_col is not None:
            y = data_values[i + seq_length, target_col]
        else:
             # Univariate case
            y = data_values[i + seq_length]
            
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)

def train_test_split_time(data, train_ratio=0.7, val_ratio=0.15):
    """
    Split data based on time (no shuffling).
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    
    return train, val, test
