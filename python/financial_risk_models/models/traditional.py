import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import logging

logger = logging.getLogger(__name__)

class TraditionalModels:
    @staticmethod
    def simulate_gbm(S0, mu, sigma, T, dt, n_sims=1):
        """
        Simulate Geometric Brownian Motion paths.
        dS_t = mu*S_t*dt + sigma*S_t*dW_t
        """
        N = int(T / dt)
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size=(n_sims, N)) 
        W = np.cumsum(W, axis=1)*np.sqrt(dt) ### Correct Brownian Motion construction
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        return t, S.T

    @staticmethod
    def fit_arima(data, order=(1, 0, 1)):
        """
        Fit ARIMA model.
        Args:
            data (pd.Series or np.array): Time series data (returns).
            order (tuple): (p, d, q) order.
        Returns:
            model_fit: Fitted ARIMA model wrapper.
        """
        try:
            model = ARIMA(data, order=order)
            model_fit = model.fit()
            return model_fit
        except Exception as e:
            logger.error(f"ARIMA fit failed: {e}")
            return None

    @staticmethod
    def fit_garch(data, p=1, q=1, mean='Zero', dist='Normal'):
        """
        Fit GARCH model.
        Args:
            data (pd.Series or np.array): Time series data (returns).
            p (int): Lag order of symmetric innovation.
            q (int): Lag order of lagged volatility.
        Returns:
            res: Fitted GARCH model result.
        """
        try:
            # GARCH(p, q) process
            # mean='Zero' is common for returns
            am = arch_model(data, vol='Garch', p=p, q=q, mean=mean, dist=dist)
            res = am.fit(disp='off')
            return res
        except Exception as e:
            logger.error(f"GARCH fit failed: {e}")
            return None
    
    @staticmethod
    def forecast_garch(model_res, horizon=1):
        """
        Forecast volatility.
        """
        forecasts = model_res.forecast(horizon=horizon)
        return forecasts.variance.iloc[-1]
