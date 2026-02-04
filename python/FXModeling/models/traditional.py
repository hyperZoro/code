"""
Traditional time series models for FX rate forecasting.
Includes ARIMA, GARCH, and Exponential Smoothing.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import product

# GARCH
from arch import arch_model

# Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class BaseModel(ABC):
    """Base class for all traditional models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.fitted = False
        self.history = None
    
    @abstractmethod
    def fit(self, data: pd.Series, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        pass
    
    def forecast(self, steps: int, alpha: float = 0.05) -> Dict:
        """
        Generate forecast with confidence intervals.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast = self.predict(steps)
        
        return {
            'mean': forecast,
            'lower': None,
            'upper': None
        }


class ARIMAModel(BaseModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.
    """
    
    def __init__(self, order: Optional[Tuple[int, int, int]] = None):
        super().__init__("ARIMA")
        self.order = order
        self.auto_select = order is None
        self.aic = None
        self.bic = None
    
    def adf_test(self, series: pd.Series) -> Dict:
        """Perform Augmented Dickey-Fuller test for stationarity."""
        result = adfuller(series.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def auto_select_order(
        self,
        data: pd.Series,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5
    ) -> Tuple[int, int, int]:
        """
        Automatically select best ARIMA order using AIC.
        """
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        # Test different differencing orders
        for d in range(max_d + 1):
            if d == 0:
                ts = data
            else:
                ts = data.diff(d).dropna()
            
            # Check if stationary after differencing
            if len(ts) > 0 and self.adf_test(ts)['is_stationary']:
                # Grid search over p and q
                for p, q in product(range(max_p + 1), range(max_q + 1)):
                    if p == 0 and q == 0:
                        continue
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except Exception as e:
                        continue
        
        print(f"Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        return best_order
    
    def fit(self, data: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5, **kwargs):
        """
        Fit ARIMA model.
        """
        # Ensure timezone-naive index to avoid statsmodels issues
        self.history = data.copy()
        if hasattr(self.history.index, 'tz') and self.history.index.tz is not None:
            self.history.index = self.history.index.tz_localize(None)
        
        if self.auto_select:
            self.order = self.auto_select_order(data, max_p, max_d, max_q)
        
        print(f"Fitting ARIMA{self.order}...")
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        self.fitted = True
        
        self.aic = self.fitted_model.aic
        self.bic = self.fitted_model.bic
        
        print(f"AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
        
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Generate point forecasts.
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values
    
    def forecast(self, steps: int, alpha: float = 0.05) -> Dict:
        """
        Generate forecast with confidence intervals.
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        mean = forecast_result.predicted_mean.values
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        return {
            'mean': mean,
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values
        }
    
    def summary(self):
        """Print model summary."""
        if self.fitted:
            print(self.fitted_model.summary())


class GARCHModel(BaseModel):
    """
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model
    for modeling volatility.
    """
    
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "Constant"
    ):
        super().__init__("GARCH")
        self.p = p
        self.q = q
        self.dist = dist  # normal, t, skewt
        self.mean = mean
        self.returns = None
    
    def fit(self, data: pd.Series, **kwargs):
        """
        Fit GARCH model on returns (not prices).
        """
        # Ensure timezone-naive index
        data = data.copy()
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Calculate returns
        self.returns = data.pct_change().dropna() * 100  # Scale for numerical stability
        self.prices = data.copy()
        self.history = data.copy()
        
        print(f"Fitting GARCH({self.p},{self.q}) with {self.dist} distribution...")
        
        self.model = arch_model(
            self.returns,
            vol='Garch',
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean=self.mean
        )
        
        self.fitted_model = self.model.fit(disp='off')
        self.fitted = True
        
        print(f"Log-likelihood: {self.fitted_model.loglikelihood:.2f}")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Predict volatility (standard deviation).
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        forecast = self.fitted_model.forecast(horizon=steps)
        # Annualized volatility
        volatility = np.sqrt(forecast.variance.values[-1]) * np.sqrt(252)
        return volatility
    
    def forecast_returns(self, steps: int) -> np.ndarray:
        """
        Forecast returns (mean prediction).
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        forecast = self.fitted_model.forecast(horizon=steps)
        # Get mean predictions and convert back from percentage
        mean_returns = forecast.mean.values[-1] / 100
        return mean_returns
    
    def forecast_prices(self, steps: int) -> np.ndarray:
        """
        Forecast prices based on return forecasts.
        """
        returns = self.forecast_returns(steps)
        last_price = self.prices.iloc[-1]
        
        prices = [last_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return np.array(prices[1:])
    
    def summary(self):
        """Print model summary."""
        if self.fitted:
            print(self.fitted_model.summary())


class ExponentialSmoothingModel(BaseModel):
    """
    Exponential Smoothing model (includes Holt-Winters).
    """
    
    def __init__(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        damped_trend: bool = False
    ):
        super().__init__("ExponentialSmoothing")
        self.trend = trend  # 'add', 'mul', None
        self.seasonal = seasonal  # 'add', 'mul', None
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
    
    def fit(self, data: pd.Series, **kwargs):
        """
        Fit Exponential Smoothing model.
        """
        # Ensure timezone-naive index
        self.history = data.copy()
        if hasattr(self.history.index, 'tz') and self.history.index.tz is not None:
            self.history.index = self.history.index.tz_localize(None)
        
        print(f"Fitting Exponential Smoothing...")
        print(f"  Trend: {self.trend}, Seasonal: {self.seasonal}")
        
        self.model = ExponentialSmoothing(
            data,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend
        )
        
        self.fitted_model = self.model.fit()
        self.fitted = True
        
        print(f"AIC: {self.fitted_model.aic:.2f}")
        print(f"BIC: {self.fitted_model.bic:.2f}")
        
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Generate point forecasts.
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        forecast = self.fitted_model.forecast(steps)
        return forecast.values
    
    def forecast(self, steps: int, alpha: float = 0.05) -> Dict:
        """
        Generate forecast with confidence intervals.
        Note: statsmodels doesn't provide CI directly, so we approximate.
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        mean = self.predict(steps)
        
        # Approximate confidence intervals using fitted values residuals
        fitted_values = self.fitted_model.fittedvalues
        residuals = self.history - fitted_values
        std_residuals = np.std(residuals)
        
        from scipy import stats
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        # Widen intervals for longer horizons
        margin = z_score * std_residuals * np.sqrt(np.arange(1, steps + 1))
        
        return {
            'mean': mean,
            'lower': mean - margin,
            'upper': mean + margin
        }
    
    def summary(self):
        """Print model summary."""
        if self.fitted:
            print(self.fitted_model.summary())


if __name__ == "__main__":
    # Test traditional models
    import sys
    sys.path.append('..')
    from data.data_loader import FXDataLoader
    
    loader = FXDataLoader()
    df = loader.fetch_fx_data(["EURUSD=X"], "2022-01-01", "2023-12-31")
    series = df['EURUSD']
    
    # Split
    train = series.iloc[:-30]
    test = series.iloc[-30:]
    
    print("=" * 50)
    print("Testing ARIMA")
    arima = ARIMAModel()
    arima.fit(train)
    arima_pred = arima.predict(len(test))
    
    print("\n" + "=" * 50)
    print("Testing GARCH")
    garch = GARCHModel(p=1, q=1)
    garch.fit(train)
    garch_pred = garch.forecast_prices(len(test))
    
    print("\n" + "=" * 50)
    print("Testing Exponential Smoothing")
    ets = ExponentialSmoothingModel(trend='add')
    ets.fit(train)
    ets_pred = ets.predict(len(test))