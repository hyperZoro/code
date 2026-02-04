import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import download_data, calculate_log_returns
from utils.backtest import Backtester
from models.deep_learning import LSTMModel, TransformerModel
from models.traditional import TraditionalModels
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Financial Risk Model Comparison...")
    
    # 1. Data Setup
    tickers = ['SPY']
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    
    data = download_data(tickers, start_date, end_date)
    if data.empty:
        logger.error("No data downloaded. Exiting.")
        return
        
    returns = calculate_log_returns(data)
    logger.info(f"Data downloaded. Returns shape: {returns.shape}")
    
    # 2. Traditional Model (GARCH)
    logger.info("Fitting Traditional Model (GARCH)...")
    # Fit on first portion just to demonstrate
    train_size = int(len(returns) * 0.7)
    train_data = returns.iloc[:train_size]
    
    # Scale returns for GARCH stability (often needed)
    garch_res = TraditionalModels.fit_garch(train_data * 100, p=1, q=1) 
    if garch_res:
        logger.info(garch_res.summary())
    else:
        logger.warning("GARCH fit failed.")

    # 3. Deep Learning Backtest
    logger.info("Starting Deep Learning Backtest (LSTM)...")
    
    # Parameters
    # Using smaller window for "verify" speed
    window_size = 500
    step_size = 250
    seq_length = 20
    
    backtester = Backtester(returns, target_col='SPY' if isinstance(returns, pd.DataFrame) else None, 
                           window_size=window_size, step_size=step_size, seq_length=seq_length)
    
    dl_config = {
        'hidden_size': 32,
        'num_layers': 1, 
        'output_size': 1,
        'dropout': 0.1
    }
    
    results = backtester.run_dl(LSTMModel, dl_config, epochs=2, batch_size=32, device='cpu')
    
    if results:
        metrics, all_preds, all_actuals = backtester.aggregate_results(results)
        logger.info(f"Backtest Complete. Overall Metrics: {metrics}")
        
        # Simple plot (saved to file)
        plt.figure(figsize=(10, 6))
        plt.plot(all_actuals, label='Actual Returns')
        plt.plot(all_preds, label='Predicted Returns (LSTM)', alpha=0.7)
        plt.legend()
        plt.title('Backtest Results: LSTM vs Actual')
        plt.savefig('backtest_result.png')
        logger.info("Plot saved to backtest_result.png")
    else:
        logger.warning("No backtest results generated (possibly insufficient data for window settings).")

if __name__ == "__main__":
    main()
