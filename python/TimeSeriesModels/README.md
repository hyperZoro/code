# GBPUSD Time Series Model Comparison

This project compares AR1 (Autoregressive Order 1) and LSTM (Long Short-Term Memory) models for forecasting GBPUSD exchange rates. The implementation follows the methodology outlined in `doc/CompareAR1withDeepLearning.md`.

## Project Overview

The project implements a comprehensive backtesting framework to compare classical statistical models (AR1) with modern deep learning approaches (LSTM) for financial time series modeling. The focus is on GBPUSD exchange rate forecasting using weekly data.

## Key Features

- **Data Fetching**: Automated download of GBPUSD data from Yahoo Finance
- **Wednesday Data Extraction**: Filters data to use only Wednesday observations to avoid holiday effects
- **AR1 Model**: Maximum likelihood estimation with simulation capabilities
- **LSTM Model**: Deep learning approach with proper data normalization and early stopping
- **Backtesting Framework**: Rolling window analysis with 50 hop dates
- **Binomial Tests**: Statistical validation at 5th, 15th, 85th, and 95th percentiles
- **Risk Metrics**: VaR and Expected Shortfall calculations
- **Comprehensive Visualization**: Multiple plots for model comparison and analysis

## Project Structure

```
TimeSeriesModels/
├── data/                          # Data storage
│   ├── gbpusd_full.csv           # Full daily data
│   ├── gbpusd_wednesday.csv      # Wednesday-only data
│   └── backtest_results.csv      # Backtesting results
├── doc/                          # Documentation
│   └── CompareAR1withDeepLearning.md
├── plots/                        # Generated plots
│   ├── model_demonstration.png
│   ├── parameter_stability.png
│   ├── risk_metrics.png
│   └── backtest_plots.png
├── results/                      # Analysis results
│   └── summary_report.txt
├── data_fetcher.py              # Data downloading module
├── ar1_model.py                 # AR1 model implementation
├── lstm_model.py                # LSTM model implementation
├── backtesting.py               # Backtesting framework
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TimeSeriesModels
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import tensorflow, pandas, numpy, matplotlib; print('All dependencies installed successfully!')"
   ```

## Usage

### Quick Start

Run the complete analysis:

```bash
python main.py
```

This will:
1. Download GBPUSD data from Yahoo Finance
2. Extract Wednesday-only observations
3. Demonstrate both models on a sample dataset
4. Run comprehensive backtesting
5. Generate performance metrics and visualizations
6. Create a detailed summary report

### Individual Components

#### Data Fetching

```python
from data_fetcher import ExchangeRateDataFetcher

# Fetch data
fetcher = ExchangeRateDataFetcher()
data = fetcher.fetch_from_yahoo_finance()
wednesday_data = fetcher.get_wednesday_data(data)
```

#### AR1 Model

```python
from ar1_model import AR1Model

# Fit model
model = AR1Model()
model.fit(data, method='mle')

# Make predictions
prediction = model.predict(last_value, steps=10)

# Run simulations
simulations = model.simulate(last_value, steps=10, n_simulations=1000)
```

#### LSTM Model

```python
from lstm_model import LSTMModel

# Fit model
model = LSTMModel(sequence_length=8, hidden_units=50)
model.fit(data, epochs=100, verbose=0)

# Make predictions
prediction = model.predict(data, steps=10)

# Run simulations
simulations = model.simulate(data, steps=10, n_simulations=1000)
```

#### Backtesting

```python
from backtesting import ModelBacktester

# Initialize backtester
backtester = ModelBacktester(data)

# Run backtest
results = backtester.run_backtest()

# Perform statistical tests
test_results = backtester.perform_binomial_tests()

# Calculate metrics
metrics = backtester.calculate_performance_metrics()
```

## Methodology

### Data Preparation
- **Source**: Yahoo Finance GBPUSD exchange rates
- **Frequency**: Daily data filtered to Wednesday observations
- **Period**: 5 years of historical data
- **Preprocessing**: No additional transformations (raw exchange rates)

### Model Specifications

#### AR1 Model
- **Equation**: X_t = c + φ * X_{t-1} + ε_t
- **Estimation**: Maximum Likelihood (equivalent to OLS under normality)
- **Parameters**: Constant (c), autoregressive coefficient (φ), residual standard deviation (σ)

#### LSTM Model
- **Architecture**: 2 LSTM layers with dropout regularization
- **Input**: 8-week sequence of historical data
- **Hidden Units**: 50 (first layer), 25 (second layer)
- **Dropout Rate**: 0.2
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error

### Backtesting Framework
- **Calibration Window**: 3 years (156 weeks)
- **Hop Dates**: 50 dates with 10-day intervals
- **Simulations**: 1000 Monte Carlo paths per prediction
- **Forecast Horizon**: 10 days ahead
- **Evaluation**: Binomial tests at 5th, 15th, 85th, 95th percentiles

## Output Files

### Data Files
- `data/gbpusd_full.csv`: Complete daily GBPUSD data
- `data/gbpusd_wednesday.csv`: Wednesday-only observations
- `data/backtest_results.csv`: Detailed backtesting results

### Visualization Files
- `plots/model_demonstration.png`: Model comparison demonstration
- `plots/parameter_stability.png`: Parameter evolution over time
- `plots/risk_metrics.png`: Risk metrics analysis
- `plots/backtest_plots.png`: Backtesting results visualization

### Reports
- `results/summary_report.txt`: Comprehensive analysis summary

## Performance Metrics

The analysis provides several performance metrics:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Binomial Test P-values**: Statistical validation at different percentiles
- **VaR**: Value at Risk (95% confidence)
- **Expected Shortfall**: Conditional Value at Risk

## Model Comparison

### AR1 Model Advantages
- **Simplicity**: Easy to interpret and implement
- **Computational Efficiency**: Fast training and prediction
- **Statistical Foundation**: Well-established theoretical properties
- **Parameter Stability**: Consistent parameter estimates

### LSTM Model Advantages
- **Non-linearity**: Captures complex temporal patterns
- **Memory**: Long-term dependency modeling
- **Flexibility**: Adaptable to various time series structures
- **Feature Learning**: Automatic feature extraction

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning utilities
- **tensorflow**: Deep learning framework
- **yfinance**: Yahoo Finance data access
- **scipy**: Scientific computing and statistics
- **statsmodels**: Statistical modeling

## Troubleshooting

### Common Issues

1. **TensorFlow Installation**: If you encounter TensorFlow installation issues, try:
   ```bash
   pip install tensorflow-cpu  # For CPU-only systems
   ```

2. **Data Fetching Errors**: If Yahoo Finance is unavailable, the code includes fallback options for other data sources.

3. **Memory Issues**: For large datasets, consider reducing the number of simulations or hop dates in the backtesting configuration.

### Performance Optimization

- **GPU Acceleration**: Install TensorFlow-GPU for faster LSTM training
- **Parallel Processing**: The backtesting framework can be extended for parallel execution
- **Data Caching**: Results are automatically saved to avoid recomputation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gbpusd_model_comparison,
  title={GBPUSD Time Series Model Comparison: AR1 vs LSTM},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/TimeSeriesModels}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This implementation is for educational and research purposes. The models should not be used for actual trading without proper validation and risk management.
