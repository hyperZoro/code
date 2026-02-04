"""
Main pipeline for FX rate modeling.

This script orchestrates the entire workflow:
1. Load FX data from Yahoo Finance
2. Feature engineering
3. Train traditional models (ARIMA, GARCH, etc.)
4. Train PyTorch models (LSTM, GRU, Transformer)
5. Evaluate and compare all models
6. Generate visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import FXDataLoader
from data.features import FeatureEngineer
from models.traditional import ARIMAModel, GARCHModel, ExponentialSmoothingModel
from models.pytorch_models import LSTMModel, GRUModel, TransformerModel, PyTorchModelWrapper
from evaluation.comparator import ModelComparator
from utils.visualization import Visualizer
from utils.helpers import load_config, save_config


class FXModelingPipeline:
    """
    Main pipeline for FX rate modeling and comparison.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)
        self.data_loader = FXDataLoader()
        self.feature_engineer = FeatureEngineer(self.config['features'])
        self.comparator = ModelComparator(self.config['evaluation']['metrics'])
        self.visualizer = Visualizer(self.config['output']['plots_dir'])
        
        self.raw_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.models = {}
        self.predictions = {}
    
    def load_data(self) -> pd.DataFrame:
        """
        Load FX data from Yahoo Finance.
        """
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        data_config = self.config['data']
        
        self.raw_data = self.data_loader.load_or_fetch(
            pairs=data_config['fx_pairs'],
            start_date=data_config['start_date'],
            end_date=data_config['end_date'],
            freq=data_config['freq']
        )
        
        print(f"\nLoaded data shape: {self.raw_data.shape}")
        print(f"Date range: {self.raw_data.index[0]} to {self.raw_data.index[-1]}")
        print(f"FX pairs: {list(self.raw_data.columns)}")
        
        return self.raw_data
    
    def prepare_data(self):
        """
        Split data and create features.
        """
        print("\n" + "=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        
        # Split data
        self.train_data, self.val_data, self.test_data = self.data_loader.split_data(
            self.raw_data,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio']
        )
        
        # Create visualizations
        self.visualizer.plot_fx_rates(self.raw_data, "FX Rates", "fx_rates.png")
        self.visualizer.plot_returns(self.raw_data, "FX Returns", "returns.png")
        self.visualizer.plot_correlation_matrix(self.raw_data, "FX Correlations", "correlations.png")
        self.visualizer.plot_volatility(self.raw_data, window=20, save_path="volatility.png")
        
        print("\nData visualization plots saved.")
    
    def train_traditional_models(self, target_col: str):
        """
        Train traditional statistical models.
        """
        print("\n" + "=" * 60)
        print("TRAINING TRADITIONAL MODELS")
        print("=" * 60)
        
        train_series = self.train_data[target_col]
        test_series = self.test_data[target_col]
        
        # ARIMA
        if self.config['traditional_models']['arima']['enabled']:
            print("\n--- Training ARIMA ---")
            arima_config = self.config['traditional_models']['arima']
            
            arima = ARIMAModel()
            arima.fit(
                train_series,
                max_p=arima_config.get('max_p', 5),
                max_d=arima_config.get('max_d', 2),
                max_q=arima_config.get('max_q', 5)
            )
            
            self.models['ARIMA'] = arima
            
            # Forecast
            for horizon in self.config['evaluation']['horizon']:
                predictions = arima.predict(horizon)
                
                # Align predictions with test data
                test_aligned = test_series.iloc[:horizon] if len(test_series) >= horizon else test_series
                pred_aligned = predictions[:len(test_aligned)]
                
                self.comparator.add_result(
                    'ARIMA',
                    test_aligned.values,
                    pred_aligned,
                    horizon
                )
            
            arima_preds = arima.predict(len(test_series))
            self.predictions['ARIMA'] = (test_series.index[:len(arima_preds)], arima_preds)
        
        # GARCH
        if self.config['traditional_models']['garch']['enabled']:
            print("\n--- Training GARCH ---")
            garch_config = self.config['traditional_models']['garch']
            
            garch = GARCHModel(
                p=garch_config.get('p', 1),
                q=garch_config.get('q', 1),
                dist=garch_config.get('dist', 'normal')
            )
            garch.fit(train_series)
            
            self.models['GARCH'] = garch
            
            # Forecast prices
            for horizon in self.config['evaluation']['horizon']:
                predictions = garch.forecast_prices(horizon)
                
                test_aligned = test_series.iloc[:horizon] if len(test_series) >= horizon else test_series
                pred_aligned = predictions[:len(test_aligned)]
                
                self.comparator.add_result(
                    'GARCH',
                    test_aligned.values,
                    pred_aligned,
                    horizon
                )
            
            garch_preds = garch.forecast_prices(len(test_series))
            self.predictions['GARCH'] = (test_series.index[:len(garch_preds)], garch_preds)
        
        # Exponential Smoothing
        if self.config['traditional_models']['exponential_smoothing']['enabled']:
            print("\n--- Training Exponential Smoothing ---")
            
            ets = ExponentialSmoothingModel(trend='add')
            ets.fit(train_series)
            
            self.models['ExpSmoothing'] = ets
            
            for horizon in self.config['evaluation']['horizon']:
                predictions = ets.predict(horizon)
                
                test_aligned = test_series.iloc[:horizon] if len(test_series) >= horizon else test_series
                pred_aligned = predictions[:len(test_aligned)]
                
                self.comparator.add_result(
                    'ExpSmoothing',
                    test_aligned.values,
                    pred_aligned,
                    horizon
                )
            
            ets_preds = ets.predict(len(test_series))
            self.predictions['ExpSmoothing'] = (test_series.index[:len(ets_preds)], ets_preds)
    
    def train_pytorch_models(self, target_col: str):
        """
        Train PyTorch deep learning models.
        """
        print("\n" + "=" * 60)
        print("TRAINING PYTORCH MODELS")
        print("=" * 60)
        
        # Prepare sequences for PyTorch models
        print("\nPreparing sequences...")
        
        # Use log returns as target for better stationarity
        train_series = self.train_data[target_col]
        val_series = self.val_data[target_col]
        test_series = self.test_data[target_col]
        
        train_returns = np.log(train_series / train_series.shift(1)).dropna()
        val_returns = np.log(val_series / val_series.shift(1)).dropna()
        test_returns = np.log(test_series / test_series.shift(1)).dropna()
        
        # Create features
        feature_cfg = self.config['features']
        feature_cfg['lags'] = list(range(1, 6))  # Add lag features
        
        train_features = self.feature_engineer.create_all_features(
            train_series.to_frame(), feature_cfg
        ).dropna()
        val_features = self.feature_engineer.create_all_features(
            val_series.to_frame(), feature_cfg
        ).dropna()
        test_features = self.feature_engineer.create_all_features(
            test_series.to_frame(), feature_cfg
        ).dropna()
        
        # Prepare sequences
        seq_length = self.config['pytorch_models']['lstm']['sequence_length']
        
        def prepare_sequences(data, target, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:(i + seq_len)])
                y.append(target[i + seq_len])
            return np.array(X), np.array(y)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        train_scaled = scaler.fit_transform(train_features)
        val_scaled = scaler.transform(val_features)
        test_scaled = scaler.transform(test_features)
        
        X_train, y_train = prepare_sequences(train_scaled, train_scaled[:, 0], seq_length)
        X_val, y_val = prepare_sequences(val_scaled, val_scaled[:, 0], seq_length)
        X_test, y_test = prepare_sequences(test_scaled, test_scaled[:, 0], seq_length)
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Validation sequences: {X_val.shape}")
        print(f"Test sequences: {X_test.shape}")
        
        # Train LSTM
        if self.config['pytorch_models']['lstm']['enabled']:
            print("\n--- Training LSTM ---")
            lstm_config = self.config['pytorch_models']['lstm']
            
            lstm = PyTorchModelWrapper(LSTMModel, X_train.shape[2], lstm_config)
            lstm.fit(X_train, y_train, X_val, y_val, verbose=True)
            
            self.models['LSTM'] = lstm
            
            # Convert predictions back to price space
            lstm_preds_scaled = lstm.predict(X_test)
            # Inverse transform - column 0 is the price
            lstm_preds_full = np.zeros((len(lstm_preds_scaled), test_scaled.shape[1]))
            lstm_preds_full[:, 0] = lstm_preds_scaled
            lstm_prices = scaler.inverse_transform(lstm_preds_full)[:, 0]
            
            # Align with test data: predictions start at test_series[seq_length]
            test_dates_aligned = test_series.iloc[seq_length:seq_length + len(lstm_prices)]
            
            for horizon in self.config['evaluation']['horizon']:
                n_points = min(horizon, len(lstm_prices))
                if n_points > 0:
                    test_aligned = test_dates_aligned.iloc[:n_points]
                    pred_aligned = lstm_prices[:n_points]
                    
                    self.comparator.add_result(
                        'LSTM',
                        test_aligned.values,
                        pred_aligned,
                        horizon
                    )
            
            # Store predictions with proper date alignment for plotting
            self.predictions['LSTM'] = (test_dates_aligned.index, lstm_prices)
            self.visualizer.plot_training_history(
                lstm.model.history, "LSTM Training History", "lstm_history.png"
            )
        
        # Train GRU
        if self.config['pytorch_models']['gru']['enabled']:
            print("\n--- Training GRU ---")
            gru_config = self.config['pytorch_models']['gru']
            
            gru = PyTorchModelWrapper(GRUModel, X_train.shape[2], gru_config)
            gru.fit(X_train, y_train, X_val, y_val, verbose=True)
            
            self.models['GRU'] = gru
            
            gru_preds_scaled = gru.predict(X_test)
            gru_preds_full = np.zeros((len(gru_preds_scaled), test_scaled.shape[1]))
            gru_preds_full[:, 0] = gru_preds_scaled
            gru_prices = scaler.inverse_transform(gru_preds_full)[:, 0]
            
            test_dates_aligned = test_series.iloc[seq_length:seq_length + len(gru_prices)]
            
            for horizon in self.config['evaluation']['horizon']:
                n_points = min(horizon, len(gru_prices))
                if n_points > 0:
                    test_aligned = test_dates_aligned.iloc[:n_points]
                    pred_aligned = gru_prices[:n_points]
                    
                    self.comparator.add_result(
                        'GRU',
                        test_aligned.values,
                        pred_aligned,
                        horizon
                    )
            
            self.predictions['GRU'] = (test_dates_aligned.index, gru_prices)
            self.visualizer.plot_training_history(
                gru.model.history, "GRU Training History", "gru_history.png"
            )
        
        # Train Transformer
        if self.config['pytorch_models']['transformer']['enabled']:
            print("\n--- Training Transformer ---")
            transformer_config = self.config['pytorch_models']['transformer']
            
            transformer = PyTorchModelWrapper(TransformerModel, X_train.shape[2], transformer_config)
            transformer.fit(X_train, y_train, X_val, y_val, verbose=True)
            
            self.models['Transformer'] = transformer
            
            trans_preds_scaled = transformer.predict(X_test)
            trans_preds_full = np.zeros((len(trans_preds_scaled), test_scaled.shape[1]))
            trans_preds_full[:, 0] = trans_preds_scaled
            trans_prices = scaler.inverse_transform(trans_preds_full)[:, 0]
            
            test_dates_aligned = test_series.iloc[seq_length:seq_length + len(trans_prices)]
            
            for horizon in self.config['evaluation']['horizon']:
                n_points = min(horizon, len(trans_prices))
                if n_points > 0:
                    test_aligned = test_dates_aligned.iloc[:n_points]
                    pred_aligned = trans_prices[:n_points]
                    
                    self.comparator.add_result(
                        'Transformer',
                        test_aligned.values,
                        pred_aligned,
                        horizon
                    )
            
            self.predictions['Transformer'] = (test_dates_aligned.index, trans_prices)
            self.visualizer.plot_training_history(
                transformer.model.history, "Transformer Training History", "transformer_history.png"
            )
    
    def evaluate_and_visualize(self, target_col: str):
        """
        Evaluate all models and generate comparison visualizations.
        """
        print("\n" + "=" * 60)
        print("EVALUATION & VISUALIZATION")
        print("=" * 60)
        
        # Print summary
        print(self.comparator.summary())
        
        # Save results
        os.makedirs(self.config['output']['results_dir'], exist_ok=True)
        results_path = os.path.join(self.config['output']['results_dir'], 'comparison_results.json')
        self.comparator.save_results(results_path)
        
        # Comparison table
        for horizon in self.config['evaluation']['horizon']:
            df = self.comparator.get_comparison_table(horizon)
            print(f"\n--- Horizon: {horizon} days ---")
            print(df)
            
            # Plot metrics comparison
            self.visualizer.plot_metrics_comparison(
                df,
                f"Metrics Comparison (Horizon: {horizon} days)",
                f"metrics_comparison_h{horizon}.png"
            )
        
        # Plot predictions comparison
        test_series = self.test_data[target_col]
        
        if self.predictions:
            # Find date range that covers all predictions
            all_dates = set()
            for dates, _ in self.predictions.values():
                all_dates.update(dates)
            common_dates = sorted(all_dates)
            
            # Reindex all predictions to common date range
            predictions_aligned = {}
            for name, (dates, values) in self.predictions.items():
                pred_series = pd.Series(values, index=dates)
                predictions_aligned[name] = pred_series.reindex(common_dates).values
            
            # Get actual values for the common date range
            actual_aligned = test_series.reindex(common_dates).values
            
            self.visualizer.plot_comparison(
                actual_aligned,
                predictions_aligned,
                dates=common_dates,
                title=f"Model Predictions Comparison - {target_col}",
                save_path="predictions_comparison.png"
            )
            
            # Create a focused comparison for the overlapping period only
            # Find dates where ALL models have predictions
            valid_dates = []
            for d in common_dates:
                has_all = True
                for name, (dates, _) in self.predictions.items():
                    if d not in dates:
                        has_all = False
                        break
                if has_all:
                    valid_dates.append(d)
            
            if len(valid_dates) > 5:
                focused_predictions = {}
                for name, (dates, values) in self.predictions.items():
                    pred_series = pd.Series(values, index=dates)
                    focused_predictions[name] = pred_series.reindex(valid_dates).values
                
                focused_actual = test_series.reindex(valid_dates).values
                
                self.visualizer.plot_comparison(
                    focused_actual,
                    focused_predictions,
                    dates=valid_dates,
                    title=f"Model Comparison - Overlapping Period ({len(valid_dates)} days)",
                    save_path="predictions_comparison_focused.png"
                )
            
            # Plot residuals for best model (using aligned data)
            best_model = self.comparator.get_best_model('rmse', 1)
            if best_model and best_model in self.predictions:
                dates, values = self.predictions[best_model]
                pred_series = pd.Series(values, index=dates)
                aligned_pred = pred_series.reindex(common_dates).values
                aligned_actual = test_series.reindex(common_dates).values
                # Remove NaN values for residual plot
                mask = ~(np.isnan(aligned_actual) | np.isnan(aligned_pred))
                if mask.any():
                    self.visualizer.plot_residuals(
                        aligned_actual[mask],
                        aligned_pred[mask],
                        title=f"Residuals - {best_model}",
                        save_path=f"residuals_{best_model}.png"
                    )
    
    def run(self, target_col: Optional[str] = None):
        """
        Run the full pipeline.
        """
        start_time = datetime.now()
        
        # Load data
        self.load_data()
        
        # Prepare data
        self.prepare_data()
        
        # Select target column
        if target_col is None:
            target_col = self.raw_data.columns[0]
        print(f"\nTarget FX pair: {target_col}")
        
        # Train models
        self.train_traditional_models(target_col)
        self.train_pytorch_models(target_col)
        
        # Evaluate
        self.evaluate_and_visualize(target_col)
        
        # Save models if configured
        if self.config['output']['save_models']:
            models_dir = os.path.join(self.config['output']['results_dir'], 'models')
            os.makedirs(models_dir, exist_ok=True)
            for name, model in self.models.items():
                if hasattr(model, 'save'):
                    model.save(os.path.join(models_dir, f"{name}.pt"))
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print(f"PIPELINE COMPLETED in {duration}")
        print("=" * 60)
        print(f"Plots saved to: {self.config['output']['plots_dir']}")
        print(f"Results saved to: {self.config['output']['results_dir']}")


def main():
    parser = argparse.ArgumentParser(description='FX Rate Modeling Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--target', type=str, default=None,
                        help='Target FX pair to model (e.g., EURUSD)')
    parser.add_argument('--pairs', type=str, nargs='+', default=None,
                        help='FX pairs to load (overrides config)')
    
    args = parser.parse_args()
    
    # Load and update config if needed
    config = load_config(args.config)
    if args.pairs:
        config['data']['fx_pairs'] = args.pairs
    
    # Run pipeline
    pipeline = FXModelingPipeline(args.config)
    pipeline.run(target_col=args.target)


if __name__ == "__main__":
    main()