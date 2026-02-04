import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from utils.data_loader import download_data, calculate_log_returns, create_sequences, train_test_split_time

def test_pipeline():
    print("Testing Pipeline...")
    tickers = ['SPY', 'EURUSD=X']
    start = '2020-01-01'
    end = '2023-01-01'
    
    # 1. Download
    print(f"Downloading {tickers}...")
    prices = download_data(tickers, start, end)
    print("Download shape:", prices.shape)
    print("Columns:", prices.columns)
    
    if prices.empty:
        print("ERROR: Downloaded dataframe is empty.")
        return

    # 2. Returns
    print("Calculating returns...")
    returns = calculate_log_returns(prices)
    print("Returns shape:", returns.shape)
    print("Head:\n", returns.head())
    
    # 3. Split
    print("Splitting data (SPY only for sequence)...")
    spy_returns = returns['SPY'].values
    train, val, test = train_test_split_time(spy_returns)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # 4. Sequences
    print("Creating sequences...")
    seq_len = 10
    X, y = create_sequences(train, seq_len)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    print("Pipeline verification complete.")

if __name__ == "__main__":
    test_pipeline()
