"""
Backtesting module for comparing AR1 and LSTM models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from data_fetcher import ExchangeRateDataFetcher
from ar1_model import AR1Model
from lstm_model import LSTMModel
from gbm_model import GBMModel
import warnings
warnings.filterwarnings('ignore')


class ModelBacktester:
    """
    Backtesting framework for comparing AR1 and LSTM models.
    """
    
    def __init__(self, data, calibration_window=156, hop_interval=10, n_hops=50, n_simulations=1000):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data with 'Date' and 'Close' columns
        calibration_window : int
            Number of weeks (3 years = 156 weeks) for model calibration
        hop_interval : int
            Number of days between hop dates
        n_hops : int
            Number of hop dates to test
        n_simulations : int
            Number of simulations per prediction
        """
        self.data = data
        self.calibration_window = calibration_window
        self.hop_interval = hop_interval
        self.n_hops = n_hops
        self.n_simulations = n_simulations
        
        # Results storage
        self.results = {
            'hop_dates': [],
            'ar1_predictions': [],
            'lstm_predictions': [],
            'gbm_predictions': [],
            'ar1_simulations': [],
            'lstm_simulations': [],
            'gbm_simulations': [],
            'realized_values': [],
            'ar1_models': [],
            'lstm_models': [],
            'gbm_models': []
        }
        
    def generate_hop_dates(self):
        """
        Generate hop dates for backtesting.
        
        Returns:
        --------
        list
            List of hop dates
        """
        # Start from the end of calibration window
        start_idx = self.calibration_window
        
        # Generate hop dates
        hop_dates = []
        current_idx = start_idx
        
        while len(hop_dates) < self.n_hops and current_idx < len(self.data) - self.hop_interval:
            hop_dates.append(current_idx)
            current_idx += self.hop_interval
        
        return hop_dates
    
    def fit_models(self, calibration_data):
        """
        Fit AR1, LSTM, and GBM models to calibration data.
        
        Parameters:
        -----------
        calibration_data : array-like
            Data for model calibration
            
        Returns:
        --------
        tuple
            (ar1_model, lstm_model, gbm_model)
        """
        # Fit AR1 model
        ar1_model = AR1Model(handle_nonstationary=True)
        ar1_model.fit(calibration_data, method='mle', force_stationary=True)
        
        # Fit LSTM model
        lstm_model = LSTMModel(sequence_length=8, hidden_units=50, dropout_rate=0.2)
        lstm_model.fit(calibration_data, epochs=100, batch_size=32, verbose=0)
        
        # Fit GBM model
        gbm_model = GBMModel(dt=1/52)  # Weekly time steps
        gbm_model.fit(calibration_data, method='mle')
        
        return ar1_model, lstm_model, gbm_model
    
    def run_backtest(self):
        """
        Run the complete backtesting procedure.
        
        Returns:
        --------
        dict
            Backtesting results
        """
        print("Starting backtesting procedure...")
        
        # Generate hop dates
        hop_dates = self.generate_hop_dates()
        print(f"Generated {len(hop_dates)} hop dates")
        
        # Extract price data
        prices = self.data['Close'].values
        
        for i, hop_idx in enumerate(hop_dates):
            print(f"Processing hop {i+1}/{len(hop_dates)} (date: {self.data.iloc[hop_idx]['Date'].strftime('%Y-%m-%d')})")
            
            # Get calibration data (3 years before hop date)
            start_idx = max(0, hop_idx - self.calibration_window)
            end_idx = hop_idx
            calibration_data = prices[start_idx:end_idx]
            
            # Get realized value (10 days after hop date)
            realized_idx = min(len(prices) - 1, hop_idx + self.hop_interval)
            realized_value = prices[realized_idx]
            
            # Fit models
            try:
                ar1_model, lstm_model, gbm_model = self.fit_models(calibration_data)
                
                # Get last value for prediction
                last_value = calibration_data[-1]
                
                # Make predictions
                ar1_pred = ar1_model.predict(last_value, steps=self.hop_interval)
                lstm_pred = lstm_model.predict(calibration_data, steps=self.hop_interval)
                gbm_pred = gbm_model.predict(last_value, steps=self.hop_interval)
                
                # Run simulations
                ar1_sims = ar1_model.simulate(last_value, steps=self.hop_interval, n_simulations=self.n_simulations)
                lstm_sims = lstm_model.simulate(calibration_data, steps=self.hop_interval, n_simulations=self.n_simulations)
                gbm_sims = gbm_model.simulate(last_value, steps=self.hop_interval, n_simulations=self.n_simulations)
                
                # Store results
                self.results['hop_dates'].append(self.data.iloc[hop_idx]['Date'])
                self.results['ar1_predictions'].append(ar1_pred[-1])  # 10-day ahead prediction
                self.results['lstm_predictions'].append(lstm_pred[-1])  # 10-day ahead prediction
                self.results['gbm_predictions'].append(gbm_pred[-1])  # 10-day ahead prediction
                self.results['ar1_simulations'].append(ar1_sims[:, -1])  # 10-day ahead simulations
                self.results['lstm_simulations'].append(lstm_sims[:, -1])  # 10-day ahead simulations
                self.results['gbm_simulations'].append(gbm_sims[:, -1])  # 10-day ahead simulations
                self.results['realized_values'].append(realized_value)
                self.results['ar1_models'].append(ar1_model)
                self.results['lstm_models'].append(lstm_model)
                self.results['gbm_models'].append(gbm_model)
                
            except Exception as e:
                print(f"Error processing hop {i+1}: {e}")
                continue
        
        print("Backtesting completed!")
        return self.results
    
    def perform_binomial_tests(self, percentiles=[5, 15, 85, 95]):
        """
        Perform binomial tests at specified percentiles.
        
        Parameters:
        -----------
        percentiles : list
            List of percentiles to test
            
        Returns:
        --------
        dict
            Test results for each model and percentile
        """
        if not self.results['hop_dates']:
            raise ValueError("No backtesting results available. Run backtest first.")
        
        test_results = {
            'ar1': {},
            'lstm': {},
            'gbm': {}
        }
        
        for percentile in percentiles:
            print(f"Performing binomial tests at {percentile}th percentile...")
            
            # AR1 model tests
            ar1_hits = []
            for i, (sims, realized) in enumerate(zip(self.results['ar1_simulations'], self.results['realized_values'])):
                threshold = np.percentile(sims, percentile)
                hit = realized <= threshold
                ar1_hits.append(hit)
            
            # LSTM model tests
            lstm_hits = []
            for i, (sims, realized) in enumerate(zip(self.results['lstm_simulations'], self.results['realized_values'])):
                threshold = np.percentile(sims, percentile)
                hit = realized <= threshold
                lstm_hits.append(hit)
            
            # GBM model tests
            gbm_hits = []
            for i, (sims, realized) in enumerate(zip(self.results['gbm_simulations'], self.results['realized_values'])):
                threshold = np.percentile(sims, percentile)
                hit = realized <= threshold
                gbm_hits.append(hit)
            
            # Perform binomial tests
            n_tests = len(ar1_hits)
            expected_prob = percentile / 100
            
            # AR1 test
            ar1_successes = sum(ar1_hits)
            ar1_test = stats.binomtest(ar1_successes, n_tests, expected_prob)
            ar1_pvalue = ar1_test.proportions_ci()[1] if ar1_successes > 0 else 1.0
            
            # LSTM test
            lstm_successes = sum(lstm_hits)
            lstm_test = stats.binomtest(lstm_successes, n_tests, expected_prob)
            lstm_pvalue = lstm_test.proportions_ci()[1] if lstm_successes > 0 else 1.0
            
            # GBM test
            gbm_successes = sum(gbm_hits)
            gbm_test = stats.binomtest(gbm_successes, n_tests, expected_prob)
            gbm_pvalue = gbm_test.proportions_ci()[1] if gbm_successes > 0 else 1.0
            
            test_results['ar1'][percentile] = {
                'hits': ar1_hits,
                'successes': ar1_successes,
                'expected': expected_prob,
                'actual_rate': ar1_successes / n_tests,
                'p_value': ar1_pvalue
            }
            
            test_results['lstm'][percentile] = {
                'hits': lstm_hits,
                'successes': lstm_successes,
                'expected': expected_prob,
                'actual_rate': lstm_successes / n_tests,
                'p_value': lstm_pvalue
            }
            
            test_results['gbm'][percentile] = {
                'hits': gbm_hits,
                'successes': gbm_successes,
                'expected': expected_prob,
                'actual_rate': gbm_successes / n_tests,
                'p_value': gbm_pvalue
            }
        
        return test_results
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for both models.
        
        Returns:
        --------
        dict
            Performance metrics
        """
        if not self.results['hop_dates']:
            raise ValueError("No backtesting results available. Run backtest first.")
        
        # Calculate prediction errors
        ar1_errors = np.array(self.results['realized_values']) - np.array(self.results['ar1_predictions'])
        lstm_errors = np.array(self.results['realized_values']) - np.array(self.results['lstm_predictions'])
        gbm_errors = np.array(self.results['realized_values']) - np.array(self.results['gbm_predictions'])
        
        # Calculate metrics
        metrics = {
            'ar1': {
                'rmse': np.sqrt(np.mean(ar1_errors**2)),
                'mae': np.mean(np.abs(ar1_errors)),
                'mape': np.mean(np.abs(ar1_errors / np.array(self.results['realized_values']))) * 100,
                'mean_error': np.mean(ar1_errors),
                'std_error': np.std(ar1_errors)
            },
            'lstm': {
                'rmse': np.sqrt(np.mean(lstm_errors**2)),
                'mae': np.mean(np.abs(lstm_errors)),
                'mape': np.mean(np.abs(lstm_errors / np.array(self.results['realized_values']))) * 100,
                'mean_error': np.mean(lstm_errors),
                'std_error': np.std(lstm_errors)
            },
            'gbm': {
                'rmse': np.sqrt(np.mean(gbm_errors**2)),
                'mae': np.mean(np.abs(gbm_errors)),
                'mape': np.mean(np.abs(gbm_errors / np.array(self.results['realized_values']))) * 100,
                'mean_error': np.mean(gbm_errors),
                'std_error': np.std(gbm_errors)
            }
        }
        
        return metrics
    
    def plot_results(self, save_path=None):
        """
        Plot backtesting results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.results['hop_dates']:
            raise ValueError("No backtesting results available. Run backtest first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Predictions vs Realized
        axes[0, 0].plot(self.results['hop_dates'], self.results['realized_values'], 'k-', label='Realized', alpha=0.7)
        axes[0, 0].plot(self.results['hop_dates'], self.results['ar1_predictions'], 'r--', label='AR1 Predictions', alpha=0.7)
        axes[0, 0].plot(self.results['hop_dates'], self.results['lstm_predictions'], 'b--', label='LSTM Predictions', alpha=0.7)
        axes[0, 0].plot(self.results['hop_dates'], self.results['gbm_predictions'], 'g--', label='GBM Predictions', alpha=0.7)
        axes[0, 0].set_title('Predictions vs Realized Values')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('GBPUSD Rate')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Prediction Errors
        ar1_errors = np.array(self.results['realized_values']) - np.array(self.results['ar1_predictions'])
        lstm_errors = np.array(self.results['realized_values']) - np.array(self.results['lstm_predictions'])
        gbm_errors = np.array(self.results['realized_values']) - np.array(self.results['gbm_predictions'])
        
        axes[0, 1].plot(self.results['hop_dates'], ar1_errors, 'r-', label='AR1 Errors', alpha=0.7)
        axes[0, 1].plot(self.results['hop_dates'], lstm_errors, 'b-', label='LSTM Errors', alpha=0.7)
        axes[0, 1].plot(self.results['hop_dates'], gbm_errors, 'g-', label='GBM Errors', alpha=0.7)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Prediction Errors')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Error Distribution
        axes[1, 0].hist(ar1_errors, bins=20, alpha=0.7, label='AR1', color='red')
        axes[1, 0].hist(lstm_errors, bins=20, alpha=0.7, label='LSTM', color='blue')
        axes[1, 0].hist(gbm_errors, bins=20, alpha=0.7, label='GBM', color='green')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Plot 4: Performance Comparison
        metrics = self.calculate_performance_metrics()
        metric_names = ['RMSE', 'MAE', 'MAPE']
        ar1_metrics = [metrics['ar1']['rmse'], metrics['ar1']['mae'], metrics['ar1']['mape']]
        lstm_metrics = [metrics['lstm']['rmse'], metrics['lstm']['mae'], metrics['lstm']['mape']]
        gbm_metrics = [metrics['gbm']['rmse'], metrics['gbm']['mae'], metrics['gbm']['mape']]
        
        x = np.arange(len(metric_names))
        width = 0.25
        
        axes[1, 1].bar(x - width, ar1_metrics, width, label='AR1', color='red', alpha=0.7)
        axes[1, 1].bar(x, lstm_metrics, width, label='LSTM', color='blue', alpha=0.7)
        axes[1, 1].bar(x + width, gbm_metrics, width, label='GBM', color='green', alpha=0.7)
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metric_names)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath):
        """
        Save backtesting results to CSV.
        
        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        if not self.results['hop_dates']:
            raise ValueError("No backtesting results available. Run backtest first.")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'hop_date': self.results['hop_dates'],
            'realized_value': self.results['realized_values'],
            'ar1_prediction': self.results['ar1_predictions'],
            'lstm_prediction': self.results['lstm_predictions'],
            'gbm_prediction': self.results['gbm_predictions'],
            'ar1_error': np.array(self.results['realized_values']) - np.array(self.results['ar1_predictions']),
            'lstm_error': np.array(self.results['realized_values']) - np.array(self.results['lstm_predictions']),
            'gbm_error': np.array(self.results['realized_values']) - np.array(self.results['gbm_predictions'])
        })
        
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")


def run_complete_backtest(data_file=None, save_results=True):
    """
    Run complete backtesting procedure.
    
    Parameters:
    -----------
    data_file : str, optional
        Path to data file. If None, will fetch data from Yahoo Finance
    save_results : bool
        Whether to save results to files
        
    Returns:
    --------
    tuple
        (backtester, test_results, metrics)
    """
    # Load or fetch data
    if data_file:
        data = pd.read_csv(data_file)
        data['Date'] = pd.to_datetime(data['Date'])
    else:
        fetcher = ExchangeRateDataFetcher()
        data = fetcher.fetch_from_yahoo_finance()
        wednesday_data = fetcher.get_wednesday_data(data)
        data = wednesday_data
    
    # Initialize backtester
    backtester = ModelBacktester(data)
    
    # Run backtest
    results = backtester.run_backtest()
    
    # Perform binomial tests
    test_results = backtester.perform_binomial_tests()
    
    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics()
    
    # Print results
    print("\n" + "="*50)
    print("BACKTESTING RESULTS")
    print("="*50)
    
    print("\nPerformance Metrics:")
    print(f"{'Metric':<10} {'AR1':<15} {'LSTM':<15} {'GBM':<15}")
    print("-" * 55)
    for metric in ['rmse', 'mae', 'mape']:
        print(f"{metric.upper():<10} {metrics['ar1'][metric]:<15.6f} {metrics['lstm'][metric]:<15.6f} {metrics['gbm'][metric]:<15.6f}")
    
    print("\nBinomial Test Results:")
    print(f"{'Model':<6} {'Percentile':<12} {'Expected':<10} {'Actual':<10} {'P-value':<10}")
    print("-" * 50)
    
    for model in ['ar1', 'lstm', 'gbm']:
        for percentile in [5, 15, 85, 95]:
            result = test_results[model][percentile]
            print(f"{model.upper():<6} {percentile:>2}th{'':<9} {result['expected']:<10.3f} "
                  f"{result['actual_rate']:<10.3f} {result['p_value']:<10.4f}")
    
    # Save results if requested
    if save_results:
        backtester.save_results("data/backtest_results.csv")
        backtester.plot_results("data/backtest_plots.png")
    
    return backtester, test_results, metrics


if __name__ == "__main__":
    # Run complete backtesting
    backtester, test_results, metrics = run_complete_backtest()
