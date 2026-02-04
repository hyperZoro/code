"""
Model comparison framework for FX rate forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime

from .metrics import MetricsCalculator


class ModelComparator:
    """
    Compare performance of multiple forecasting models.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ['rmse', 'mae', 'mape', 'directional_accuracy']
        self.results = {}
        self.predictions = {}
    
    def add_result(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon: int = 1
    ):
        """
        Add a model's results.
        """
        # Store predictions
        key = f"{model_name}_h{horizon}"
        self.predictions[key] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'horizon': horizon
        }
        
        # Calculate metrics
        all_metrics = MetricsCalculator.calculate_all(y_true, y_pred)
        
        # Filter requested metrics
        result = {m: all_metrics.get(m, float('nan')) for m in self.metrics}
        result['horizon'] = horizon
        
        if model_name not in self.results:
            self.results[model_name] = {}
        
        self.results[model_name][horizon] = result
    
    def get_comparison_table(self, horizon: Optional[int] = None) -> pd.DataFrame:
        """
        Get comparison table of all models.
        """
        rows = []
        
        for model_name, horizons in self.results.items():
            if horizon is not None:
                if horizon in horizons:
                    row = {'model': model_name, **horizons[horizon]}
                    rows.append(row)
            else:
                for h, metrics in horizons.items():
                    row = {'model': model_name, **metrics}
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index('model')
        
        return df
    
    def get_best_model(self, metric: str = 'rmse', horizon: int = 1) -> str:
        """
        Get the best performing model for a given metric.
        """
        df = self.get_comparison_table(horizon)
        
        if df.empty:
            return None
        
        # For directional accuracy, higher is better
        ascending = metric not in ['directional_accuracy', 'r2']
        
        best_idx = df[metric].sort_values(ascending=ascending).index[0]
        return best_idx
    
    def get_model_rankings(self, metric: str = 'rmse', horizon: int = 1) -> pd.DataFrame:
        """
        Get model rankings for a specific metric.
        """
        df = self.get_comparison_table(horizon)
        
        if df.empty:
            return pd.DataFrame()
        
        # For directional accuracy, higher is better
        ascending = metric not in ['directional_accuracy', 'r2']
        
        df_ranked = df[[metric]].sort_values(metric, ascending=ascending)
        df_ranked['rank'] = range(1, len(df_ranked) + 1)
        
        return df_ranked
    
    def compare_horizons(self, model_name: str) -> pd.DataFrame:
        """
        Compare a model's performance across different horizons.
        """
        if model_name not in self.results:
            return pd.DataFrame()
        
        rows = []
        for horizon, metrics in self.results[model_name].items():
            rows.append({'horizon': horizon, **metrics})
        
        return pd.DataFrame(rows).set_index('horizon')
    
    def summary(self) -> str:
        """
        Get a text summary of all results.
        """
        lines = ["=" * 60]
        lines.append("MODEL COMPARISON SUMMARY")
        lines.append("=" * 60)
        
        for horizon in sorted(set(
            h for horizons in self.results.values() for h in horizons.keys()
        )):
            lines.append(f"\n--- Horizon: {horizon} days ---")
            df = self.get_comparison_table(horizon)
            lines.append(df.to_string())
            
            best = self.get_best_model('rmse', horizon)
            lines.append(f"\nBest model (RMSE): {best}")
        
        return "\n".join(lines)
    
    def save_results(self, path: str):
        """
        Save comparison results to JSON.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for model, horizons in self.results.items():
            results_serializable[model] = {
                str(h): {k: float(v) for k, v in metrics.items()}
                for h, metrics in horizons.items()
            }
        
        with open(path, 'w') as f:
            json.dump({
                'results': results_serializable,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Results saved to {path}")
    
    def load_results(self, path: str):
        """
        Load comparison results from JSON.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.metrics = data['metrics']
        self.results = {
            model: {int(h): metrics for h, metrics in horizons.items()}
            for model, horizons in data['results'].items()
        }
        
        print(f"Results loaded from {path}")
    
    def get_predictions_df(self, model_name: str, horizon: int = 1) -> pd.DataFrame:
        """
        Get predictions as a DataFrame.
        """
        key = f"{model_name}_h{horizon}"
        if key not in self.predictions:
            return pd.DataFrame()
        
        pred = self.predictions[key]
        return pd.DataFrame({
            'actual': pred['y_true'],
            'predicted': pred['y_pred'],
            'error': pred['y_true'] - pred['y_pred']
        })


if __name__ == "__main__":
    # Test comparator
    np.random.seed(42)
    y_true = np.random.randn(100) + 100
    
    comparator = ModelComparator()
    
    # Add some dummy results
    for model in ['ARIMA', 'LSTM', 'GRU']:
        for horizon in [1, 5, 10]:
            noise = np.random.randn(100) * (0.5 + horizon * 0.1)
            y_pred = y_true + noise
            comparator.add_result(model, y_true, y_pred, horizon)
    
    print(comparator.summary())
    print("\n" + "=" * 60)
    print("Model Rankings (RMSE, horizon=1):")
    print(comparator.get_model_rankings('rmse', 1))