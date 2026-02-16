

# Implementation Plan: FX Trade Translation Backtesting

## 1. Project Configuration & Parameters

Define a centralized `Config` class or dictionary to manage the backtest settings.

* **Asset:** FX Pair (default `EURUSD=X`).
* **Timeframe:** Total historical window (e.g., 2018–2024).
* **Backtest Settings:** * `num_hops`: **Configurable integer** (e.g., 20).
* `hop_interval_days`: Horizon for each simulation (e.g., 21 days).
* `sim_paths`: 1,000 (as specified).


* **Calibration Settings:** 156-week window (Wednesdays only).

## 2. Data Acquisition Module (`data_loader.py`)

This module will handle API calls and data cleaning.

* **Spot FX:** Use `yfinance` to download daily OHLC data.
* **Interest Rates:** Attempt to fetch 3-month Treasury rates (US) and EURIBOR (or equivalent) via `pandas_datareader` (FRED).
* *Fallback:* If remote fetch fails, use a scalar constant (e.g., ).


* **Implied Volatility:** Attempt to fetch `^EVZ` (Euro Volatility Index).
* *Fallback:* Calculate a 30-day rolling realized volatility as a proxy for ATM Implied Vol if ticker data is unavailable.


* **Preprocessing:** Filter the main dataframe to create a "Wednesday-only" series for the GBM calibration.

## 3. Quantitative Core (`analytics.py`)

Implementation of the mathematical engines.

* **GBM Calibration:** * Input: 156-period Wednesday spot series.
* Output: Annualized drift () and volatility () based on log-returns.


* **Pricers:**
* **Forward:** 
* **Option:** Garman-Kohlhagen (GK) Black-Scholes variation for FX.


* **Monte Carlo Engine:** * Generate 1,000 paths for FX Spot at  using the calibrated GBM.

## 4. Trade Translation Logic (`translator.py`)

This is the heart of the experiment. We define two methods to translate a "Trade as of Today" () to a "Historical Hop Date" ().

| Method | Logic | Formula |
| --- | --- | --- |
| **A: Spot Ratio** | Keep Moneyness constant |  |
| **B: Vol-Adjusted** | Keep "Standardized" distance | , where  |

## 5. Backtest Execution Loop (`engine.py`)

1. **Date Selection:** Identify the most recent date as . Calculate  hop dates backward from today, spaced by the `hop_interval`.
2. **The Loop:** For each hop date :
* **Step 1:** Isolate the 3y Wednesday window prior to .
* **Step 2:** Calibrate GBM parameters ().
* **Step 3:** Translate the "current" trade to  using **Method A** and **Method B**.
* **Step 4:** Run 1,000 simulations from  to .
* **Step 5:** Calculate the "Realized Value" using the actual market spot at .
* **Step 6:** Compute the **PIT Value**:





## 6. Evaluation & Visualization (`reporting.py`)

* **Histograms:** Plot the distribution of PIT values for Method A and Method B.
* *Interpretation:* A perfectly calibrated model and translation will result in a **Uniform Distribution**.


* **Statistics:** Calculate the Kolmogorov-Smirnov (KS) test score to see which method deviates less from uniformity.

