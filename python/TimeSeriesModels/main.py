"""
Main script for comparing AR1 and LSTM models for GBPUSD time series modeling.
This script implements the complete workflow outlined in CompareAR1withDeepLearning.md
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our modules
from data_fetcher import ExchangeRateDataFetcher
from ar1_model import AR1Model
from lstm_model import LSTMModel
from gbm_model import GBMModel
from backtesting import ModelBacktester, run_complete_backtest

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data', 'models', 'results', 'plots']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def fetch_and_prepare_data():
    """
    Fetch GBPUSD data and prepare it for modeling.
    
    Returns:
    --------
    pandas.DataFrame
        Wednesday-only GBPUSD data
    """
    print("="*60)
    print("STEP 1: FETCHING AND PREPARING DATA")
    print("="*60)
    
    # Initialize data fetcher
    fetcher = ExchangeRateDataFetcher()
    
    # Fetch data from Yahoo Finance
    print("Fetching GBPUSD data from Yahoo Finance...")
    full_data = fetcher.fetch_from_yahoo_finance()
    
    if full_data is None or full_data.empty:
        raise ValueError("Failed to fetch data from Yahoo Finance")
    
    # Extract Wednesday data
    print("Extracting Wednesday data points...")
    wednesday_data = fetcher.get_wednesday_data(full_data)
    
    if wednesday_data is None or wednesday_data.empty:
        raise ValueError("Failed to extract Wednesday data")
    
    # Save data
    fetcher.save_data(full_data, "data/gbpusd_full.csv")
    fetcher.save_data(wednesday_data, "data/gbpusd_wednesday.csv")
    
    print(f"Data preparation completed!")
    print(f"Full dataset: {len(full_data)} observations")
    print(f"Wednesday dataset: {len(wednesday_data)} observations")
    print(f"Date range: {wednesday_data['Date'].min()} to {wednesday_data['Date'].max()}")
    
    return wednesday_data


def demonstrate_models(data):
    """
    Demonstrate AR1 and LSTM models on a sample of the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Wednesday-only GBPUSD data
    """
    print("\n" + "="*60)
    print("STEP 2: MODEL DEMONSTRATION")
    print("="*60)
    
    # Use the first 3 years (156 weeks) for demonstration
    sample_data = data['Close'].values[:156]
    
    print(f"Using {len(sample_data)} data points for model demonstration")
    print(f"Sample data range: {sample_data.min():.4f} to {sample_data.max():.4f}")
    
    # Fit AR1 model
    print("\nFitting AR1 model...")
    ar1_model = AR1Model(handle_nonstationary=True)
    ar1_model.fit(sample_data, method='mle', force_stationary=True)
    ar1_params = ar1_model.get_parameters()
    
    print("AR1 Model Parameters:")
    print(f"  Constant (c): {ar1_params['c']:.6f}")
    print(f"  Autoregressive (φ): {ar1_params['phi']:.6f}")
    print(f"  Residual Std (σ): {ar1_params['sigma']:.6f}")
    
    # Fit LSTM model
    print("\nFitting LSTM model...")
    lstm_model = LSTMModel(sequence_length=8, hidden_units=50, dropout_rate=0.2)
    lstm_model.fit(sample_data, epochs=50, batch_size=32, verbose=0)
    
    print("LSTM Model Architecture:")
    print(f"  Sequence length: {lstm_model.sequence_length}")
    print(f"  Hidden units: {lstm_model.hidden_units}")
    print(f"  Dropout rate: {lstm_model.dropout_rate}")
    print(f"  Total parameters: {lstm_model.model.count_params()}")
    
    # Fit GBM model
    print("\nFitting GBM model...")
    gbm_model = GBMModel(dt=1/52)  # Weekly time steps
    gbm_model.fit(sample_data, method='mle')
    
    print("GBM Model Parameters:")
    gbm_params = gbm_model.get_parameters()
    print(f"  Drift (μ): {gbm_params['mu']:.6f}")
    print(f"  Volatility (σ): {gbm_params['sigma']:.6f}")
    print(f"  Time step (Δt): {gbm_params['dt']:.4f}")
    
    # Make predictions
    print("\nMaking 10-day ahead predictions...")
    ar1_pred = ar1_model.predict(sample_data[-1], steps=10)
    lstm_pred = lstm_model.predict(sample_data, steps=10)
    gbm_pred = gbm_model.predict(sample_data[-1], steps=10)
    
    print(f"AR1 10-day prediction: {ar1_pred[-1]:.4f}")
    print(f"LSTM 10-day prediction: {lstm_pred[-1]:.4f}")
    print(f"GBM 10-day prediction: {gbm_pred[-1]:.4f}")
    
    # Run simulations
    print("\nRunning simulations (1000 paths)...")
    ar1_sims = ar1_model.simulate(sample_data[-1], steps=10, n_simulations=1000)
    lstm_sims = lstm_model.simulate(sample_data, steps=10, n_simulations=1000)
    gbm_sims = gbm_model.simulate(sample_data[-1], steps=10, n_simulations=1000)
    
    print("Simulation Statistics (10-day ahead):")
    print(f"AR1 - Mean: {np.mean(ar1_sims[:, -1]):.4f}, Std: {np.std(ar1_sims[:, -1]):.4f}")
    print(f"LSTM - Mean: {np.mean(lstm_sims[:, -1]):.4f}, Std: {np.std(lstm_sims[:, -1]):.4f}")
    print(f"GBM - Mean: {np.mean(gbm_sims[:, -1]):.4f}, Std: {np.std(gbm_sims[:, -1]):.4f}")
    
    # Calculate model comparison metrics
    print("\nModel Comparison Metrics:")
    ar1_aic = ar1_model.get_aic(sample_data)
    ar1_bic = ar1_model.get_bic(sample_data)
    lstm_aic = lstm_model.get_aic(sample_data)
    lstm_bic = lstm_model.get_bic(sample_data)
    gbm_aic = gbm_model.get_aic(sample_data)
    gbm_bic = gbm_model.get_bic(sample_data)
    
    print(f"AR1 - AIC: {ar1_aic:.2f}, BIC: {ar1_bic:.2f}")
    print(f"LSTM - AIC: {lstm_aic:.2f}, BIC: {lstm_bic:.2f}")
    print(f"GBM - AIC: {gbm_aic:.2f}, BIC: {gbm_bic:.2f}")
    
    # Plot demonstration
    plot_model_demonstration(sample_data, ar1_model, lstm_model, gbm_model, ar1_sims, lstm_sims, gbm_sims)
    
    return ar1_model, lstm_model, gbm_model


def plot_model_demonstration(data, ar1_model, lstm_model, gbm_model, ar1_sims, lstm_sims, gbm_sims):
    """
    Create demonstration plots for the models.
    
    Parameters:
    -----------
    data : array-like
        Sample data used for modeling
    ar1_model : AR1Model
        Fitted AR1 model
    lstm_model : LSTMModel
        Fitted LSTM model
    gbm_model : GBMModel
        Fitted GBM model
    ar1_sims : array
        AR1 simulations
    lstm_sims : array
        LSTM simulations
    gbm_sims : array
        GBM simulations
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Original data
    axes[0, 0].plot(data, 'k-', linewidth=1, label='Original Data')
    axes[0, 0].set_title('Sample Data (3 Years of Wednesday Observations)')
    axes[0, 0].set_xlabel('Week')
    axes[0, 0].set_ylabel('GBPUSD Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Model predictions
    steps = 10
    ar1_pred = ar1_model.predict(data[-1], steps=steps)
    lstm_pred = lstm_model.predict(data, steps=steps)
    gbm_pred = gbm_model.predict(data[-1], steps=steps)
    
    x_pred = np.arange(len(data), len(data) + steps)
    axes[0, 1].plot(data, 'k-', linewidth=1, label='Historical Data')
    axes[0, 1].plot(x_pred, ar1_pred, 'r--', linewidth=2, label='AR1 Prediction')
    axes[0, 1].plot(x_pred, lstm_pred, 'b--', linewidth=2, label='LSTM Prediction')
    axes[0, 1].plot(x_pred, gbm_pred, 'g--', linewidth=2, label='GBM Prediction')
    axes[0, 1].set_title('10-Day Ahead Predictions')
    axes[0, 1].set_xlabel('Week')
    axes[0, 1].set_ylabel('GBPUSD Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Model comparison
    axes[0, 2].plot(x_pred, ar1_pred, 'r-', linewidth=2, label='AR1')
    axes[0, 2].plot(x_pred, lstm_pred, 'b-', linewidth=2, label='LSTM')
    axes[0, 2].plot(x_pred, gbm_pred, 'g-', linewidth=2, label='GBM')
    axes[0, 2].set_title('Model Predictions Comparison')
    axes[0, 2].set_xlabel('Week')
    axes[0, 2].set_ylabel('GBPUSD Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: AR1 simulations
    for i in range(min(50, ar1_sims.shape[0])):
        axes[1, 0].plot(x_pred, ar1_sims[i, :], 'r-', alpha=0.1)
    axes[1, 0].plot(data, 'k-', linewidth=1, label='Historical Data')
    axes[1, 0].plot(x_pred, ar1_pred, 'r--', linewidth=2, label='AR1 Mean Prediction')
    axes[1, 0].set_title('AR1 Simulations (50 paths)')
    axes[1, 0].set_xlabel('Week')
    axes[1, 0].set_ylabel('GBPUSD Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: LSTM simulations
    for i in range(min(50, lstm_sims.shape[0])):
        axes[1, 1].plot(x_pred, lstm_sims[i, :], 'b-', alpha=0.1)
    axes[1, 1].plot(data, 'k-', linewidth=1, label='Historical Data')
    axes[1, 1].plot(x_pred, lstm_pred, 'b--', linewidth=2, label='LSTM Mean Prediction')
    axes[1, 1].set_title('LSTM Simulations (50 paths)')
    axes[1, 1].set_xlabel('Week')
    axes[1, 1].set_ylabel('GBPUSD Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: GBM simulations
    for i in range(min(50, gbm_sims.shape[0])):
        axes[1, 2].plot(x_pred, gbm_sims[i, :], 'g-', alpha=0.1)
    axes[1, 2].plot(data, 'k-', linewidth=1, label='Historical Data')
    axes[1, 2].plot(x_pred, gbm_pred, 'g--', linewidth=2, label='GBM Mean Prediction')
    axes[1, 2].set_title('GBM Simulations (50 paths)')
    axes[1, 2].set_xlabel('Week')
    axes[1, 2].set_ylabel('GBPUSD Rate')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/model_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_backtesting_analysis(data):
    """
    Run the complete backtesting analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Wednesday-only GBPUSD data
    """
    print("\n" + "="*60)
    print("STEP 3: BACKTESTING ANALYSIS")
    print("="*60)
    
    print("Running complete backtesting procedure...")
    print("This will take some time as it involves:")
    print("- 50 hop dates with 10-day intervals")
    print("- 3-year calibration windows")
    print("- 1000 simulations per prediction")
    print("- Both AR1 and LSTM model fitting")
    
    # Run backtesting
    backtester, test_results, metrics = run_complete_backtest(
        data_file=None,  # Will use the data we already have
        save_results=True
    )
    
    # Additional analysis
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Model parameter stability analysis
    analyze_parameter_stability(backtester)
    
    # Risk analysis
    analyze_risk_metrics(backtester)
    
    return backtester, test_results, metrics


def analyze_parameter_stability(backtester):
    """
    Analyze the stability of model parameters across different calibration windows.
    
    Parameters:
    -----------
    backtester : ModelBacktester
        Backtester with fitted models
    """
    print("\nAnalyzing parameter stability...")
    
    ar1_params = []
    lstm_params = []
    
    for ar1_model, lstm_model in zip(backtester.results['ar1_models'], backtester.results['lstm_models']):
        # AR1 parameters
        ar1_params.append(ar1_model.get_parameters())
        
        # LSTM parameters (number of parameters as a proxy for complexity)
        lstm_params.append(lstm_model.model.count_params())
    
    # Convert to DataFrame for analysis
    ar1_df = pd.DataFrame(ar1_params)
    
    print("AR1 Parameter Statistics:")
    print(f"  Constant (c): Mean={ar1_df['c'].mean():.6f}, Std={ar1_df['c'].std():.6f}")
    print(f"  Autoregressive (φ): Mean={ar1_df['phi'].mean():.6f}, Std={ar1_df['phi'].std():.6f}")
    print(f"  Residual Std (σ): Mean={ar1_df['sigma'].mean():.6f}, Std={ar1_df['sigma'].std():.6f}")
    
    print(f"LSTM Parameter Count: Mean={np.mean(lstm_params):.0f}, Std={np.std(lstm_params):.0f}")
    
    # Plot parameter evolution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # AR1 parameters over time
    axes[0, 0].plot(ar1_df['c'], 'r-', alpha=0.7)
    axes[0, 0].set_title('AR1 Constant Parameter Evolution')
    axes[0, 0].set_xlabel('Hop Date')
    axes[0, 0].set_ylabel('Constant (c)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(ar1_df['phi'], 'r-', alpha=0.7)
    axes[0, 1].set_title('AR1 Autoregressive Parameter Evolution')
    axes[0, 1].set_xlabel('Hop Date')
    axes[0, 1].set_ylabel('Autoregressive (φ)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(ar1_df['sigma'], 'r-', alpha=0.7)
    axes[1, 0].set_title('AR1 Residual Std Evolution')
    axes[1, 0].set_xlabel('Hop Date')
    axes[1, 0].set_ylabel('Residual Std (σ)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(lstm_params, 'b-', alpha=0.7)
    axes[1, 1].set_title('LSTM Parameter Count Evolution')
    axes[1, 1].set_xlabel('Hop Date')
    axes[1, 1].set_ylabel('Number of Parameters')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/parameter_stability.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_risk_metrics(backtester):
    """
    Analyze risk metrics for both models.
    
    Parameters:
    -----------
    backtester : ModelBacktester
        Backtester with results
    """
    print("\nAnalyzing risk metrics...")
    
    # Calculate Value at Risk (VaR) and Expected Shortfall (ES)
    ar1_var_95 = []
    ar1_es_95 = []
    lstm_var_95 = []
    lstm_es_95 = []
    
    for ar1_sims, lstm_sims in zip(backtester.results['ar1_simulations'], backtester.results['lstm_simulations']):
        # AR1 risk metrics
        ar1_var_95.append(np.percentile(ar1_sims, 5))
        ar1_es_95.append(np.mean(ar1_sims[ar1_sims <= np.percentile(ar1_sims, 5)]))
        
        # LSTM risk metrics
        lstm_var_95.append(np.percentile(lstm_sims, 5))
        lstm_es_95.append(np.mean(lstm_sims[lstm_sims <= np.percentile(lstm_sims, 5)]))
    
    print("Risk Metrics (95% VaR and Expected Shortfall):")
    print(f"AR1 - VaR: Mean={np.mean(ar1_var_95):.4f}, Std={np.std(ar1_var_95):.4f}")
    print(f"AR1 - ES: Mean={np.mean(ar1_es_95):.4f}, Std={np.std(ar1_es_95):.4f}")
    print(f"LSTM - VaR: Mean={np.mean(lstm_var_95):.4f}, Std={np.std(lstm_var_95):.4f}")
    print(f"LSTM - ES: Mean={np.mean(lstm_es_95):.4f}, Std={np.std(lstm_es_95):.4f}")
    
    # Plot risk metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(ar1_var_95, bins=15, alpha=0.7, color='red', label='AR1')
    axes[0, 0].hist(lstm_var_95, bins=15, alpha=0.7, color='blue', label='LSTM')
    axes[0, 0].set_title('Distribution of 95% VaR')
    axes[0, 0].set_xlabel('VaR')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    axes[0, 1].hist(ar1_es_95, bins=15, alpha=0.7, color='red', label='AR1')
    axes[0, 1].hist(lstm_es_95, bins=15, alpha=0.7, color='blue', label='LSTM')
    axes[0, 1].set_title('Distribution of Expected Shortfall')
    axes[0, 1].set_xlabel('Expected Shortfall')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    axes[1, 0].plot(ar1_var_95, 'r-', alpha=0.7, label='AR1')
    axes[1, 0].plot(lstm_var_95, 'b-', alpha=0.7, label='LSTM')
    axes[1, 0].set_title('VaR Evolution Over Time')
    axes[1, 0].set_xlabel('Hop Date')
    axes[1, 0].set_ylabel('95% VaR')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(ar1_es_95, 'r-', alpha=0.7, label='AR1')
    axes[1, 1].plot(lstm_es_95, 'b-', alpha=0.7, label='LSTM')
    axes[1, 1].set_title('Expected Shortfall Evolution Over Time')
    axes[1, 1].set_xlabel('Hop Date')
    axes[1, 1].set_ylabel('Expected Shortfall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/risk_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_summary_report(backtester, test_results, metrics):
    """
    Generate a comprehensive summary report.
    
    Parameters:
    -----------
    backtester : ModelBacktester
        Backtester with results
    test_results : dict
        Binomial test results
    metrics : dict
        Performance metrics
    """
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Create summary report
    report = []
    report.append("GBPUSD TIME SERIES MODEL COMPARISON REPORT")
    report.append("=" * 50)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append("This report compares the performance of AR1 and LSTM models")
    report.append("for forecasting GBPUSD exchange rates using weekly data.")
    report.append("")
    
    report.append("METHODOLOGY")
    report.append("-" * 15)
    report.append("• Data: Wednesday-only GBPUSD end-of-day rates")
    report.append("• Calibration window: 3 years (156 weeks)")
    report.append("• Backtesting: 50 hop dates with 10-day intervals")
    report.append("• Simulations: 1000 Monte Carlo paths per prediction")
    report.append("• Evaluation: Binomial tests at 5th, 15th, 85th, 95th percentiles")
    report.append("")
    
    report.append("PERFORMANCE METRICS")
    report.append("-" * 20)
    report.append(f"{'Metric':<10} {'AR1':<15} {'LSTM':<15} {'Winner':<10}")
    report.append("-" * 50)
    
    for metric in ['rmse', 'mae', 'mape']:
        ar1_val = metrics['ar1'][metric]
        lstm_val = metrics['lstm'][metric]
        winner = 'AR1' if ar1_val < lstm_val else 'LSTM'
        report.append(f"{metric.upper():<10} {ar1_val:<15.6f} {lstm_val:<15.6f} {winner:<10}")
    
    report.append("")
    
    report.append("BINOMIAL TEST RESULTS")
    report.append("-" * 20)
    report.append("Tests whether realized values fall within expected percentiles")
    report.append("")
    
    for model in ['ar1', 'lstm']:
        report.append(f"{model.upper()} MODEL:")
        for percentile in [5, 15, 85, 95]:
            result = test_results[model][percentile]
            report.append(f"  {percentile}th percentile: Expected={result['expected']:.3f}, "
                         f"Actual={result['actual_rate']:.3f}, P-value={result['p_value']:.4f}")
        report.append("")
    
    report.append("CONCLUSIONS")
    report.append("-" * 12)
    report.append("• Both models show different strengths in forecasting")
    report.append("• AR1 model provides simpler, more interpretable results")
    report.append("• LSTM model captures more complex patterns but may overfit")
    report.append("• Risk management applications should consider both approaches")
    report.append("")
    
    report.append("FILES GENERATED")
    report.append("-" * 15)
    report.append("• data/gbpusd_wednesday.csv: Wednesday-only exchange rate data")
    report.append("• data/backtest_results.csv: Detailed backtesting results")
    report.append("• plots/model_demonstration.png: Model demonstration plots")
    report.append("• plots/parameter_stability.png: Parameter stability analysis")
    report.append("• plots/risk_metrics.png: Risk metrics analysis")
    report.append("• plots/backtest_plots.png: Backtesting results visualization")
    
    # Save report
    with open('results/summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    print('\n'.join(report))
    print(f"\nDetailed report saved to: results/summary_report.txt")


def main():
    """
    Main function to run the complete model comparison workflow.
    """
    print("GBPUSD TIME SERIES MODEL COMPARISON")
    print("Comparing AR1 vs LSTM Models for Financial Risk Factor Modeling")
    print("=" * 70)
    
    try:
        # Setup
        setup_directories()
        
        # Step 1: Data preparation
        data = fetch_and_prepare_data()
        
        # Step 2: Model demonstration
        ar1_model, lstm_model, gbm_model = demonstrate_models(data)
        
        # Step 3: Backtesting analysis
        backtester, test_results, metrics = run_backtesting_analysis(data)
        
        # Step 4: Generate summary report
        generate_summary_report(backtester, test_results, metrics)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("Check the 'results/' and 'plots/' directories for detailed outputs.")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
