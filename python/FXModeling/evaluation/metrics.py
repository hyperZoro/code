"""
Evaluation metrics for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MetricsCalculator:
    """
    Calculator for various time series forecasting metrics.
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional Accuracy - percentage of correct directional predictions.
        Important for trading strategies.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Calculate actual and predicted directions
        true_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        # Count correct predictions
        correct = np.sum(true_direction == pred_direction)
        total = len(true_direction)
        
        return (correct / total) * 100 if total > 0 else 0.0
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score."""
        from sklearn.metrics import r2_score
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Theil's U statistic (inequality coefficient).
        0 = perfect forecast, 1 = no better than naive forecast.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))
        denominator = np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2))
        return numerator / denominator if denominator != 0 else float('inf')
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available metrics.
        """
        return {
            'mse': MetricsCalculator.mse(y_true, y_pred),
            'rmse': MetricsCalculator.rmse(y_true, y_pred),
            'mae': MetricsCalculator.mae(y_true, y_pred),
            'mape': MetricsCalculator.mape(y_true, y_pred),
            'smape': MetricsCalculator.smape(y_true, y_pred),
            'directional_accuracy': MetricsCalculator.directional_accuracy(y_true, y_pred),
            'r2': MetricsCalculator.r2_score(y_true, y_pred),
            'theil_u': MetricsCalculator.theil_u(y_true, y_pred)
        }
    
    @staticmethod
    def calculate_returns_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        initial_value: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate metrics based on returns (useful for trading evaluation).
        """
        # Calculate returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Cumulative returns
        true_cumret = initial_value * np.cumprod(1 + true_returns)
        pred_cumret = initial_value * np.cumprod(1 + pred_returns)
        
        # Sharpe ratio approximation
        sharpe_true = np.mean(true_returns) / (np.std(true_returns) + 1e-8) * np.sqrt(252)
        sharpe_pred = np.mean(pred_returns) / (np.std(pred_returns) + 1e-8) * np.sqrt(252)
        
        # Maximum drawdown
        peak = np.maximum.accumulate(true_cumret)
        drawdown = (true_cumret - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'sharpe_ratio': sharpe_true,
            'predicted_sharpe': sharpe_pred,
            'max_drawdown': max_drawdown * 100,  # as percentage
            'total_return': (true_cumret[-1] / initial_value - 1) * 100
        }


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    y_true = np.random.randn(100) + 100
    y_pred = y_true + np.random.randn(100) * 0.5
    
    metrics = MetricsCalculator.calculate_all(y_true, y_pred)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return_metrics = MetricsCalculator.calculate_returns_metrics(y_true, y_pred)
    print("\nReturns Metrics:")
    for k, v in return_metrics.items():
        print(f"  {k}: {v:.4f}")