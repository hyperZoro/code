import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from .data_loader import create_sequences
from .training import train_dl_model
from .evaluation import calculate_metrics, kupiec_pof_test, ks_test_uniformity
from models.deep_learning import FinancialTimeSeriesDataset

class Backtester:
    def __init__(self, data, target_col='SPY', window_size=252*2, step_size=252, seq_length=10):
        """
        Args:
            data (pd.DataFrame): Time series data (returns).
            window_size (int): Size of training window (e.g., 2 years = 504 days).
            step_size (int): Size of test window (e.g., 1 year = 252 days).
            seq_length (int): Sequence length for DL models.
        """
        self.data = data
        self.target_col = target_col
        self.window_size = window_size
        self.step_size = step_size
        self.seq_length = seq_length
        
    def run_dl(self, model_class, model_params, epochs=10, batch_size=32, device='cpu'):
        """
        Run rolling window backtest for Deep Learning models.
        """
        n = len(self.data)
        results = []
        
        # Determine split points
        # Start at window_size, step by step_size
        indices = range(self.window_size, n - self.step_size, self.step_size)
        
        target_data = self.data[self.target_col].values if isinstance(self.data, pd.DataFrame) else self.data
        
        for t in tqdm(indices, desc="Backtesting"):
            train_end = t
            test_end = t + self.step_size
            
            # Prepare data
            train_raw = target_data[t - self.window_size : t]
            test_raw = target_data[t : test_end]
            
            # Create sequences
            X_train, y_train = create_sequences(train_raw, self.seq_length)
            X_test, y_test = create_sequences(test_raw, self.seq_length)
            
            # Skip if not enough data
            if len(X_train) < batch_size or len(X_test) == 0:
                continue
                
            # Create DataLoaders
            train_dataset = FinancialTimeSeriesDataset(X_train, y_train)
            # Use part of train for validation in early stopping
            val_split = int(len(train_dataset) * 0.8)
            val_dataset = FinancialTimeSeriesDataset(X_train[val_split:], y_train[val_split:])
            train_subset = FinancialTimeSeriesDataset(X_train[:val_split], y_train[:val_split])
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            input_size = X_train.shape[2] # (seq, feature)
            model = model_class(input_size=input_size, **model_params)
            
            # Train
            model, _ = train_dl_model(model, train_loader, val_loader, epochs=epochs, device=device, patience=3)
            
            # Predict
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                preds = model(X_test_tensor).cpu().numpy().flatten()
                
            # Collect results
            # y_test are actuals
            y_actual = y_test.flatten()
            
            metrics = calculate_metrics(y_actual, preds)
            results.append({
                'window_start': t,
                'window_end': test_end,
                'metrics': metrics,
                'predictions': preds,
                'actuals': y_actual
            })
            
        return results

    def aggregate_results(self, results):
        """
        Aggregate results from all windows.
        """
        all_preds = np.concatenate([r['predictions'] for r in results])
        all_actuals = np.concatenate([r['actuals'] for r in results])
        
        overall_metrics = calculate_metrics(all_actuals, all_preds)
        
        # VaR Analysis (assuming normal distribution of errors for simplest parametric VaR, 
        # or if model predicts volatility directly. For now, let's assume model predicts returns
        # and we compute historical VaR of residuals or similar?
        # Better: DL model should predict volatility.
        # But for 'comparison', if we stick to return prediction (mean), we can check RMSE.
        # If user wants risk modelling, usually GARCH predicts variance.
        # DL for risk usually means predicting squared returns or vol.
        
        return overall_metrics, all_preds, all_actuals
