"""
Geometric Brownian Motion (GBM) Model implementation for time series modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class GBMModel:
    """
    Geometric Brownian Motion (GBM) Model implementation.
    
    The model is: dS_t = μ * S_t * dt + σ * S_t * dW_t
    where S_t is the asset price at time t, μ is the drift, σ is the volatility,
    and W_t is a Wiener process.
    
    In discrete time: S_{t+1} = S_t * exp((μ - 0.5*σ²) * Δt + σ * √Δt * Z)
    where Z ~ N(0,1) and Δt is the time step.
    """
    
    def __init__(self, dt=1/52):  # Default to weekly time steps
        """
        Initialize GBM model.
        
        Parameters:
        -----------
        dt : float
            Time step size (default: 1/52 for weekly data)
        """
        self.mu = None  # drift parameter
        self.sigma = None  # volatility parameter
        self.dt = dt
        self.fitted = False
        
    def fit(self, data, method='mle'):
        """
        Fit GBM model to data using Maximum Likelihood Estimation.
        
        Parameters:
        -----------
        data : array-like
            Time series data (asset prices)
        method : str
            Estimation method ('mle' for Maximum Likelihood, 'mom' for Method of Moments)
            
        Returns:
        --------
        self : GBMModel
            Fitted model instance
        """
        data = np.array(data)
        n = len(data)
        
        if n < 2:
            raise ValueError("Need at least 2 data points for GBM estimation")
        
        # Ensure all data points are positive
        if np.any(data <= 0):
            raise ValueError("GBM model requires positive asset prices")
        
        if method == 'mle':
            # Maximum Likelihood Estimation
            self._fit_mle(data)
        elif method == 'mom':
            # Method of Moments
            self._fit_mom(data)
        else:
            raise ValueError("Method must be 'mle' or 'mom'")
            
        self.fitted = True
        return self
    
    def _fit_mle(self, data):
        """
        Fit GBM model using Maximum Likelihood Estimation.
        """
        # Calculate log returns
        log_returns = np.diff(np.log(data))
        
        # MLE estimates for normal distribution
        mu_hat = np.mean(log_returns) / self.dt
        sigma_hat = np.sqrt(np.var(log_returns) / self.dt)
        
        self.mu = mu_hat
        self.sigma = sigma_hat
        
        # Ensure volatility is positive
        if self.sigma <= 0:
            self.sigma = 0.01  # Small positive value
    
    def _fit_mom(self, data):
        """
        Fit GBM model using Method of Moments.
        """
        # Calculate log returns
        log_returns = np.diff(np.log(data))
        
        # Method of moments estimates
        mu_hat = np.mean(log_returns) / self.dt
        sigma_hat = np.sqrt(np.var(log_returns) / self.dt)
        
        self.mu = mu_hat
        self.sigma = sigma_hat
        
        # Ensure volatility is positive
        if self.sigma <= 0:
            self.sigma = 0.01  # Small positive value
    
    def predict(self, last_value, steps=1):
        """
        Predict future values using the fitted GBM model.
        
        Parameters:
        -----------
        last_value : float
            Last observed value in the series
        steps : int
            Number of steps ahead to predict
            
        Returns:
        --------
        array
            Predicted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if last_value <= 0:
            raise ValueError("GBM model requires positive starting value")
        
        predictions = []
        current_value = last_value
        
        for _ in range(steps):
            # GBM prediction: S_{t+1} = S_t * exp((μ - 0.5*σ²) * Δt)
            drift_term = (self.mu - 0.5 * self.sigma**2) * self.dt
            predicted = current_value * np.exp(drift_term)
            predictions.append(predicted)
            current_value = predicted
        
        return np.array(predictions)
    
    def simulate(self, last_value, steps=1, n_simulations=1000):
        """
        Simulate future values using the fitted GBM model.
        
        Parameters:
        -----------
        last_value : float
            Last observed value in the series
        steps : int
            Number of steps ahead to simulate
        n_simulations : int
            Number of simulation paths
            
        Returns:
        --------
        array
            Simulated values with shape (n_simulations, steps)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulating")
        
        if last_value <= 0:
            raise ValueError("GBM model requires positive starting value")
        
        simulations = np.zeros((n_simulations, steps))
        
        for i in range(n_simulations):
            current_value = last_value
            
            for j in range(steps):
                # GBM simulation: S_{t+1} = S_t * exp((μ - 0.5*σ²) * Δt + σ * √Δt * Z)
                drift_term = (self.mu - 0.5 * self.sigma**2) * self.dt
                diffusion_term = self.sigma * np.sqrt(self.dt) * np.random.normal(0, 1)
                simulated = current_value * np.exp(drift_term + diffusion_term)
                simulations[i, j] = simulated
                current_value = simulated
        
        return simulations
    
    def get_parameters(self):
        """
        Get fitted model parameters.
        
        Returns:
        --------
        dict
            Dictionary containing model parameters
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        return {
            'mu': self.mu,
            'sigma': self.sigma,
            'dt': self.dt
        }
    
    def get_residuals(self, data):
        """
        Calculate residuals for given data.
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        array
            Residuals (log returns minus expected log returns)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating residuals")
        
        data = np.array(data)
        
        # Calculate log returns
        log_returns = np.diff(np.log(data))
        
        # Expected log returns under GBM
        expected_log_returns = (self.mu - 0.5 * self.sigma**2) * self.dt
        
        # Calculate residuals
        residuals = log_returns - expected_log_returns
        
        return residuals
    
    def get_aic(self, data):
        """
        Calculate Akaike Information Criterion (AIC).
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        float
            AIC value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating AIC")
        
        residuals = self.get_residuals(data)
        n = len(residuals)
        k = 2  # Number of parameters (mu, sigma)
        
        # Calculate log-likelihood (assuming normal residuals)
        residual_std = np.std(residuals)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * residual_std**2) - 0.5 * np.sum(residuals**2) / residual_std**2
        
        # Calculate AIC
        aic = 2 * k - 2 * log_likelihood
        
        return aic
    
    def get_bic(self, data):
        """
        Calculate Bayesian Information Criterion (BIC).
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        float
            BIC value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating BIC")
        
        residuals = self.get_residuals(data)
        n = len(residuals)
        k = 2  # Number of parameters (mu, sigma)
        
        # Calculate log-likelihood (assuming normal residuals)
        residual_std = np.std(residuals)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * residual_std**2) - 0.5 * np.sum(residuals**2) / residual_std**2
        
        # Calculate BIC
        bic = k * np.log(n) - 2 * log_likelihood
        
        return bic
    
    def get_volatility_forecast(self, horizon=1):
        """
        Get volatility forecast for given horizon.
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon in time steps
            
        Returns:
        --------
        float
            Volatility forecast
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting volatility")
        
        return self.sigma * np.sqrt(horizon * self.dt)
    
    def get_value_at_risk(self, confidence_level=0.05, horizon=1, initial_value=1.0):
        """
        Calculate Value at Risk (VaR) for given confidence level and horizon.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (e.g., 0.05 for 95% VaR)
        horizon : int
            Time horizon in time steps
        initial_value : float
            Initial asset value
            
        Returns:
        --------
        float
            Value at Risk
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating VaR")
        
        # VaR calculation for GBM
        z_score = stats.norm.ppf(confidence_level)
        drift_term = (self.mu - 0.5 * self.sigma**2) * horizon * self.dt
        volatility_term = self.sigma * np.sqrt(horizon * self.dt) * z_score
        
        var = initial_value * (np.exp(drift_term + volatility_term) - 1)
        
        return var
    
    def get_expected_shortfall(self, confidence_level=0.05, horizon=1, initial_value=1.0):
        """
        Calculate Expected Shortfall (ES) for given confidence level and horizon.
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (e.g., 0.05 for 95% ES)
        horizon : int
            Time horizon in time steps
        initial_value : float
            Initial asset value
            
        Returns:
        --------
        float
            Expected Shortfall
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating Expected Shortfall")
        
        # ES calculation for GBM
        z_score = stats.norm.ppf(confidence_level)
        phi_z = stats.norm.pdf(z_score)
        
        drift_term = (self.mu - 0.5 * self.sigma**2) * horizon * self.dt
        volatility_term = self.sigma * np.sqrt(horizon * self.dt)
        
        es = initial_value * (np.exp(drift_term) * phi_z / confidence_level - 1)
        
        return es


def fit_gbm_model(data, dt=1/52, method='mle'):
    """
    Convenience function to fit GBM model.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    dt : float
        Time step size
    method : str
        Estimation method ('mle' or 'mom')
        
    Returns:
    --------
    GBMModel
        Fitted GBM model
    """
    model = GBMModel(dt=dt)
    return model.fit(data, method=method)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample GBM data
    n = 100
    mu_true = 0.1  # 10% annual drift
    sigma_true = 0.2  # 20% annual volatility
    dt = 1/52  # weekly time steps
    
    data = np.zeros(n)
    data[0] = 1.0  # Initial price
    
    for i in range(1, n):
        drift = (mu_true - 0.5 * sigma_true**2) * dt
        diffusion = sigma_true * np.sqrt(dt) * np.random.normal(0, 1)
        data[i] = data[i-1] * np.exp(drift + diffusion)
    
    # Fit GBM model
    model = fit_gbm_model(data, dt=dt)
    
    # Print parameters
    params = model.get_parameters()
    print("Fitted GBM Parameters:")
    print(f"μ (drift): {params['mu']:.4f}")
    print(f"σ (volatility): {params['sigma']:.4f}")
    print(f"Δt (time step): {params['dt']:.4f}")
    
    # Make prediction
    prediction = model.predict(data[-1], steps=5)
    print(f"\n5-step ahead prediction: {prediction}")
    
    # Simulate
    simulations = model.simulate(data[-1], steps=5, n_simulations=100)
    print(f"\nSimulation mean: {np.mean(simulations, axis=0)}")
    print(f"Simulation std: {np.std(simulations, axis=0)}")
    
    # Risk metrics
    var_95 = model.get_value_at_risk(confidence_level=0.05, horizon=1, initial_value=data[-1])
    es_95 = model.get_expected_shortfall(confidence_level=0.05, horizon=1, initial_value=data[-1])
    print(f"\n95% VaR: {var_95:.4f}")
    print(f"95% Expected Shortfall: {es_95:.4f}")
