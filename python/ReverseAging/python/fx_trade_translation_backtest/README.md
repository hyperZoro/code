# FX Trade Translation Backtesting

Implements the workflow described in `doc/implemenation_plan.md`:
- Configurable FX backtest parameters
- Data loading with fallbacks for rates and implied vol
- 156-week Wednesday GBM calibration
- Trade translation methods A/B
- Monte Carlo PIT backtest
- Histogram + KS statistics reporting

## Run

```bash
python -m python.fx_trade_translation_backtest.main
```

Outputs are written to:
- `python/fx_trade_translation_backtest/output/backtest_summary.csv`
- `python/fx_trade_translation_backtest/output/translated_trades_by_hop.csv`
- `python/fx_trade_translation_backtest/output/portfolio_definition.csv`
- `python/fx_trade_translation_backtest/output/ks_stats.csv`
- `python/fx_trade_translation_backtest/output/pit_histograms.png`
- `python/fx_trade_translation_backtest/output/pit_hop_comparison.png`

## Portfolio Definition

The current backtest portfolio is a two-leg FX options portfolio (long call + long put) defined in `python/fx_trade_translation_backtest/config.py`:
- `portfolio_structure`: `"straddle"` or `"strangle"`
- `position_side`: `"long"` or `"short"`
- `position_units`: position multiplier per leg
- `trade_maturity_days`: tenor in days
- `trade_moneyness`: shared strike moneyness used by `straddle` (if leg overrides are `None`)
- `trade_strike_override`: shared strike override used by `straddle` (if provided)
- `call_moneyness` and `put_moneyness`: used by `strangle` when leg overrides are `None`
- `call_strike_override` and `put_strike_override`: explicit leg strikes (highest priority for each leg)

Each run writes the exact instantiated portfolio (as-of spot/vol/strike and metric notes) to:
- `python/fx_trade_translation_backtest/output/portfolio_definition.csv`

Important interpretation:
- PIT is currently computed as CDF rank of total portfolio value (`pit_metric=portfolio_value_cdf`).
- For a long call + long put portfolio, value is non-monotonic in spot, so Method A/B PIT should generally diverge when translations differ.
- If you run an ATM `straddle` (`trade_moneyness=1.0` with no strike overrides), Method A and B can still collapse to similar/identical strikes.

## Quick Portfolio Switch

Use `python/fx_trade_translation_backtest/config.py`:

- `strangle` (default):
  - `portfolio_structure = "strangle"`
  - `call_moneyness = 1.05`
  - `put_moneyness = 0.95`
- `straddle`:
  - `portfolio_structure = "straddle"`
  - `trade_moneyness = 1.0` (ATM) or a non-1.0 value for non-ATM straddle
  - optional: set `trade_strike_override` to force an explicit shared strike

## Notebook

Interactive notebook for experimentation:
- `python/fx_trade_translation_backtest/playground_visualization.ipynb`
