"""
Demo script showing how to use the GBPUSD time series modeling system.
This script demonstrates the key functionality without running the full backtesting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data_fetcher import ExchangeRateDataFetcher
from ar1_model import AR1Model
from lstm_model import LSTMModel
from gbm_model import GBMModel

def demo_data_fetching():
    """Demonstrate data fetching functionality."""
    print("="*50)
    print("DEMO: DATA FETCHING")
    print("="*50)
    
    # Initialize data fetcher
    fetcher = ExchangeRateDataFetcher()
    
    # Fetch data (using a shorter period for demo)
    print("Fetching GBPUSD data from Yahoo Finance...")
    data = fetcher.fetch_from_yahoo_finance(
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    if data is not None:
        print(f"✓ Successfully fetched {len(data)} data points")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        print(f"Price range: {data['Close'].min():.4f} to {data['Close'].max():.4f}")
        
        # Extract Wednesday data
        wednesday_data = fetcher.get_wednesday_data(data)
        print(f"✓ Extracted {len(wednesday_data)} Wednesday observations")
        
        return wednesday_data
    else:
        print("✗ Failed to fetch data")
        return None

def demo_ar1_model(data):
    """Demonstrate AR1 model functionality."""
    print("\n" + "="*50)
    print("DEMO: AR1 MODEL")
    print("="*50)
    
    # Use the first 100 observations for demo
    sample_data = data['Close'].values[:100]
    
    # Fit AR1 model
    print("Fitting AR1 model...")
    ar1_model = AR1Model(handle_nonstationary=True)
    ar1_model.fit(sample_data, method='mle', force_stationary=True)
    
    # Get parameters
    params = ar1_model.get_parameters()
    print("AR1 Model Parameters:")
    print(f"  AR coefficient (φ): {params['phi']:.6f}")
    print(f"  Residual std (σ): {params['sigma']:.6f}")
    print(f"  Is differenced: {params['is_differenced']}")
    
    # Make prediction
    print("\nMaking 5-step ahead prediction...")
    prediction = ar1_model.predict(sample_data[-1], steps=5)
    print(f"Predictions: {prediction}")
    
    # Run simulations
    print("\nRunning simulations (100 paths)...")
    simulations = ar1_model.simulate(sample_data[-1], steps=5, n_simulations=100)
    print(f"Simulation mean: {np.mean(simulations, axis=0)}")
    print(f"Simulation std: {np.std(simulations, axis=0)}")
    
    return ar1_model, prediction, simulations

def demo_gbm_model(data):
    """Demonstrate GBM model functionality."""
    print("\n" + "="*50)
    print("DEMO: GBM MODEL")
    print("="*50)
    
    # Use the first 100 observations for demo
    sample_data = data['Close'].values[:100]
    
    # Fit GBM model
    print("Fitting GBM model...")
    gbm_model = GBMModel(dt=1/52)  # Weekly time steps
    gbm_model.fit(sample_data, method='mle')
    
    # Get parameters
    params = gbm_model.get_parameters()
    print("GBM Model Parameters:")
    print(f"  Drift (μ): {params['mu']:.6f}")
    print(f"  Volatility (σ): {params['sigma']:.6f}")
    print(f"  Time step (Δt): {params['dt']:.4f}")
    
    # Make prediction
    print("\nMaking 5-step ahead prediction...")
    prediction = gbm_model.predict(sample_data[-1], steps=5)
    print(f"Predictions: {prediction}")
    
    # Run simulations
    print("\nRunning simulations (100 paths)...")
    simulations = gbm_model.simulate(sample_data[-1], steps=5, n_simulations=100)
    print(f"Simulation mean: {np.mean(simulations, axis=0)}")
    print(f"Simulation std: {np.std(simulations, axis=0)}")
    
    # Risk metrics
    var_95 = gbm_model.get_value_at_risk(confidence_level=0.05, horizon=1, initial_value=sample_data[-1])
    es_95 = gbm_model.get_expected_shortfall(confidence_level=0.05, horizon=1, initial_value=sample_data[-1])
    print(f"\n95% VaR: {var_95:.4f}")
    print(f"95% Expected Shortfall: {es_95:.4f}")
    
    return gbm_model, prediction, simulations

def demo_lstm_model(data):
    """Demonstrate LSTM model functionality."""
    print("\n" + "="*50)
    print("DEMO: LSTM MODEL")
    print("="*50)
    
    # Use the first 100 observations for demo
    sample_data = data['Close'].values[:100]
    
    # Fit LSTM model
    print("Fitting LSTM model...")
    lstm_model = LSTMModel(sequence_length=8, hidden_units=30, dropout_rate=0.2)
    lstm_model.fit(sample_data, epochs=50, batch_size=16, verbose=0)
    
    print("LSTM Model Architecture:")
    print(f"  Sequence length: {lstm_model.sequence_length}")
    print(f"  Hidden units: {lstm_model.hidden_units}")
    print(f"  Total parameters: {lstm_model.model.count_params()}")
    
    # Make prediction
    print("\nMaking 5-step ahead prediction...")
    prediction = lstm_model.predict(sample_data, steps=5)
    print(f"Predictions: {prediction}")
    
    # Run simulations
    print("\nRunning simulations (100 paths)...")
    simulations = lstm_model.simulate(sample_data, steps=5, n_simulations=100)
    print(f"Simulation mean: {np.mean(simulations, axis=0)}")
    print(f"Simulation std: {np.std(simulations, axis=0)}")
    
    return lstm_model, prediction, simulations

def demo_model_comparison(ar1_model, lstm_model, gbm_model, data):
    """Demonstrate model comparison."""
    print("\n" + "="*50)
    print("DEMO: MODEL COMPARISON")
    print("="*50)
    
    sample_data = data['Close'].values[:100]
    
    # Calculate model metrics
    print("Model Comparison Metrics:")
    
    # AR1 metrics
    ar1_aic = ar1_model.get_aic(sample_data)
    ar1_bic = ar1_model.get_bic(sample_data)
    print(f"AR1 - AIC: {ar1_aic:.2f}, BIC: {ar1_bic:.2f}")
    
    # LSTM metrics
    lstm_aic = lstm_model.get_aic(sample_data)
    lstm_bic = lstm_model.get_bic(sample_data)
    print(f"LSTM - AIC: {lstm_aic:.2f}, BIC: {lstm_bic:.2f}")
    
    # GBM metrics
    gbm_aic = gbm_model.get_aic(sample_data)
    gbm_bic = gbm_model.get_bic(sample_data)
    print(f"GBM - AIC: {gbm_aic:.2f}, BIC: {gbm_bic:.2f}")
    
    # Residuals comparison
    ar1_residuals = ar1_model.get_residuals(sample_data)
    lstm_residuals = lstm_model.get_residuals(sample_data)
    gbm_residuals = gbm_model.get_residuals(sample_data)
    
    print(f"\nResidual Statistics:")
    print(f"AR1 - Mean: {np.mean(ar1_residuals):.6f}, Std: {np.std(ar1_residuals):.6f}")
    print(f"LSTM - Mean: {np.mean(lstm_residuals):.6f}, Std: {np.std(lstm_residuals):.6f}")
    print(f"GBM - Mean: {np.mean(gbm_residuals):.6f}, Std: {np.std(gbm_residuals):.6f}")
    
    # Plot comparison
    plot_model_comparison(sample_data, ar1_model, lstm_model, gbm_model)

def plot_model_comparison(data, ar1_model, lstm_model, gbm_model):
    """Create comparison plots for the models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Original data
    axes[0, 0].plot(data, 'k-', linewidth=1, label='Original Data')
    axes[0, 0].set_title('Sample Data (100 Observations)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('GBPUSD Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Model predictions
    steps = 5
    ar1_pred = ar1_model.predict(data[-1], steps=steps)
    lstm_pred = lstm_model.predict(data, steps=steps)
    gbm_pred = gbm_model.predict(data[-1], steps=steps)
    
    x_pred = np.arange(len(data), len(data) + steps)
    axes[0, 1].plot(data, 'k-', linewidth=1, label='Historical Data')
    axes[0, 1].plot(x_pred, ar1_pred, 'r--', linewidth=2, label='AR1 Prediction')
    axes[0, 1].plot(x_pred, lstm_pred, 'b--', linewidth=2, label='LSTM Prediction')
    axes[0, 1].plot(x_pred, gbm_pred, 'g--', linewidth=2, label='GBM Prediction')
    axes[0, 1].set_title('5-Step Ahead Predictions')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('GBPUSD Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Model comparison
    axes[0, 2].plot(x_pred, ar1_pred, 'r-', linewidth=2, label='AR1')
    axes[0, 2].plot(x_pred, lstm_pred, 'b-', linewidth=2, label='LSTM')
    axes[0, 2].plot(x_pred, gbm_pred, 'g-', linewidth=2, label='GBM')
    axes[0, 2].set_title('Model Predictions Comparison')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('GBPUSD Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Residuals comparison
    ar1_residuals = ar1_model.get_residuals(data)
    lstm_residuals = lstm_model.get_residuals(data)
    gbm_residuals = gbm_model.get_residuals(data)
    
    axes[1, 0].hist(ar1_residuals, bins=15, alpha=0.7, color='red', label='AR1')
    axes[1, 0].hist(lstm_residuals, bins=15, alpha=0.7, color='blue', label='LSTM')
    axes[1, 0].hist(gbm_residuals, bins=15, alpha=0.7, color='green', label='GBM')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].set_xlabel('Residual')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Plot 5: Residuals over time
    axes[1, 1].plot(ar1_residuals, 'r-', alpha=0.7, label='AR1')
    axes[1, 1].plot(lstm_residuals, 'b-', alpha=0.7, label='LSTM')
    axes[1, 1].plot(gbm_residuals, 'g-', alpha=0.7, label='GBM')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Residuals Over Time')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Model metrics comparison
    ar1_aic = ar1_model.get_aic(data)
    lstm_aic = lstm_model.get_aic(data)
    gbm_aic = gbm_model.get_aic(data)
    
    models = ['AR1', 'LSTM', 'GBM']
    aic_values = [ar1_aic, lstm_aic, gbm_aic]
    
    axes[1, 2].bar(models, aic_values, color=['red', 'blue', 'green'], alpha=0.7)
    axes[1, 2].set_title('AIC Comparison (Lower is Better)')
    axes[1, 2].set_ylabel('AIC Value')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Comparison plot saved as 'demo_comparison.png'")

def main():
    """Run the complete demo."""
    print("GBPUSD TIME SERIES MODELING DEMO")
    print("="*50)
    print("This demo shows the key functionality of the system.")
    print("For full backtesting analysis, run: python main.py")
    print()
    
    try:
        # Step 1: Data fetching
        data = demo_data_fetching()
        if data is None:
            print("Demo failed: Could not fetch data")
            return
        
        # Step 2: AR1 model
        ar1_model, ar1_pred, ar1_sims = demo_ar1_model(data)
        
        # Step 3: LSTM model
        lstm_model, lstm_pred, lstm_sims = demo_lstm_model(data)
        
        # Step 4: GBM model
        gbm_model, gbm_pred, gbm_sims = demo_gbm_model(data)
        
        # Step 5: Model comparison
        demo_model_comparison(ar1_model, lstm_model, gbm_model, data)
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Key takeaways:")
        print("• All three models can be fitted to GBPUSD data")
        print("• AR1 provides simple, interpretable results")
        print("• LSTM captures more complex patterns")
        print("• GBM is well-suited for financial asset modeling")
        print("• Use main.py for comprehensive backtesting")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
