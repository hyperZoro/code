# FX Rate Modeling Project

A comprehensive framework for modeling foreign exchange (FX) rates using both traditional statistical models and deep learning approaches with PyTorch.

## Features

### Data Sources
- **Yahoo Finance** integration for free FX rate data
- Automatic caching to avoid repeated API calls
- Support for major FX pairs (EUR/USD, GBP/USD, USD/JPY, etc.)

### Traditional Models
- **ARIMA** - AutoRegressive Integrated Moving Average with automatic order selection
- **GARCH** - Generalized Autoregressive Conditional Heteroskedasticity for volatility modeling
- **Exponential Smoothing** - Holt-Winters method with trend and seasonality support

### Deep Learning Models (PyTorch)
- **LSTM** - Long Short-Term Memory networks
- **GRU** - Gated Recurrent Unit networks
- **Transformer** - Attention-based sequence models

### Evaluation & Comparison
- Multiple metrics: RMSE, MAE, MAPE, Directional Accuracy, R²
- Model comparison across different prediction horizons
- Comprehensive visualizations

## Project Structure

```
FXModeling/
├── configs/
│   └── config.yaml          # Main configuration file
├── data/
│   ├── data_loader.py       # Yahoo Finance data loading
│   └── features.py          # Feature engineering utilities
├── models/
│   ├── traditional.py       # ARIMA, GARCH, Exponential Smoothing
│   └── pytorch_models.py    # LSTM, GRU, Transformer
├── evaluation/
│   ├── metrics.py           # Evaluation metrics
│   └── comparator.py        # Model comparison framework
├── utils/
│   ├── visualization.py     # Plotting utilities
│   └── helpers.py           # Helper functions
├── pipeline.py              # Main pipeline orchestration
├── run.py                   # Simple run script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
cd FXModeling
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Full Pipeline

```bash
python run.py
```

Or with command-line options:
```bash
python pipeline.py --target EURUSD --pairs EURUSD=X GBPUSD=X USDJPY=X
```

### Using as a Library

```python
from data.data_loader import FXDataLoader
from data.features import FeatureEngineer
from models.traditional import ARIMAModel
from models.pytorch_models import LSTMModel, PyTorchModelWrapper

# Load data
loader = FXDataLoader()
df = loader.fetch_fx_data(
    pairs=["EURUSD=X", "GBPUSD=X"],
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# Train ARIMA
arima = ARIMAModel()
arima.fit(df['EURUSD'])
forecast = arima.predict(steps=30)

# Train LSTM
# (see pipeline.py for full example)
```

## Configuration

Edit `configs/config.yaml` to customize:

- **FX pairs** to analyze
- **Date range** for training/testing
- **Feature engineering** options
- **Model hyperparameters**
- **Evaluation metrics**

### Example Configuration

```yaml
data:
  fx_pairs:
    - EURUSD=X
    - GBPUSD=X
    - USDJPY=X
  start_date: "2020-01-01"
  train_ratio: 0.8

pytorch_models:
  lstm:
    hidden_size: 128
    num_layers: 2
    epochs: 100
    learning_rate: 0.001
```

## Output

The pipeline generates:

- **Plots**: FX rates, returns, volatility, model predictions, comparison charts
- **Results**: JSON files with model performance metrics
- **Models**: Saved model checkpoints (optional)

All outputs are saved to:
- `./plots/` - Visualizations
- `./results/` - Metrics and model files

## Key Features

### Feature Engineering
- Returns (simple and log)
- Rolling volatility
- Simple/Exponential moving averages
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Lag features

### Evaluation Metrics
- **MSE/RMSE/MAE** - Standard error metrics
- **MAPE/sMAPE** - Percentage errors
- **Directional Accuracy** - Important for trading decisions
- **R²** - Coefficient of determination
- **Theil's U** - Forecast quality measure

## Model Comparison

The framework automatically compares:
- Performance across different prediction horizons (1, 5, 10, 20 days)
- Traditional vs. deep learning approaches
- Statistical significance of differences

## Notes

- Yahoo Finance data is free but has rate limits
- Deep learning models require more training time
- GPU acceleration is automatically used if available
- Models are compared on a held-out test set

## License

MIT License - Feel free to use and modify for your purposes.