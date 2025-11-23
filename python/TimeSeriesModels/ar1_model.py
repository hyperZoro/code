"""
Autoregressive Order 1 (AR1) Model implementation for time series modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')


class AR1Model:
    """
    Autoregressive Order 1 (AR1) Model implementation.
    
    The model is: X_t = φ * X_{t-1} + ε_t
    where φ is the autoregressive parameter and ε_t ~ N(0, σ²)
    """
    
    def __init__(self, handle_nonstationary=True):
        """
        Initialize AR1 model.
        
        Parameters:
        -----------
        handle_nonstationary : bool
            Whether to automatically handle non-stationary data
        """
        self.phi = None  # autoregressive parameter
        self.sigma = None  # error standard deviation
        self.fitted = False
        self.handle_nonstationary = handle_nonstationary
        self.is_differenced = False
        self.original_data = None
        self.differenced_data = None
        
    def _test_stationarity(self, data):
        """
        Test for stationarity using ADF and KPSS tests.
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        dict
            Test results
        """
        # Augmented Dickey-Fuller test
        adf_result = adfuller(data)
        
        # KPSS test
        kpss_result = kpss(data, regression='c')
        
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_critical_values': kpss_result[3]
        }
    
    def _is_stationary(self, data):
        """
        Determine if data is stationary based on ADF and KPSS tests.
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        bool
            True if stationary, False otherwise
        """
        tests = self._test_stationarity(data)
        
        # ADF: H0 is non-stationary, reject if p-value < 0.05
        adf_stationary = tests['adf_pvalue'] < 0.05
        
        # KPSS: H0 is stationary, reject if p-value < 0.05
        kpss_stationary = tests['kpss_pvalue'] >= 0.05
        
        return adf_stationary and kpss_stationary
    
    def _make_stationary(self, data):
        """
        Make data stationary using differencing.
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        tuple
            (stationary_data, differencing_order)
        """
        original_data = np.array(data)
        current_data = original_data.copy()
        order = 0
        
        while order < 2:  # Limit to 2nd order differencing
            if self._is_stationary(current_data):
                break
            
            # First difference
            current_data = np.diff(current_data)
            order += 1
            
            if len(current_data) < 10:  # Need minimum observations
                break
        
        return current_data, order
    
    def _remove_trend(self, data):
        """
        Remove linear trend from data.
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        tuple
            (detrended_data, trend_coefficients)
        """
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        detrended = data - trend
        
        return detrended, coeffs
    
    def fit(self, data, method='mle', force_stationary=False):
        """
        Fit AR1 model to data using specified method.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        method : str
            Estimation method ('mle', 'ols', 'yule_walker')
        force_stationary : bool
            Force stationarity by differencing if needed
            
        Returns:
        --------
        self : AR1Model
            Fitted model instance
        """
        data = np.array(data)
        self.original_data = data.copy()
        
        # Handle non-stationarity if requested
        if self.handle_nonstationary and force_stationary:
            print("Testing for stationarity...")
            tests = self._test_stationarity(data)
            print(f"ADF p-value: {tests['adf_pvalue']:.4f}")
            print(f"KPSS p-value: {tests['kpss_pvalue']:.4f}")
            
            if not self._is_stationary(data):
                print("Data is non-stationary. Applying differencing...")
                stationary_data, order = self._make_stationary(data)
                self.differenced_data = stationary_data
                self.is_differenced = True
                print(f"Applied {order} order differencing")
                
                # Re-test stationarity
                if self._is_stationary(stationary_data):
                    print("Data is now stationary after differencing")
                else:
                    print("Warning: Data still appears non-stationary")
                
                data = stationary_data
            else:
                print("Data is stationary")
        
        n = len(data)
        
        if n < 2:
            raise ValueError("Need at least 2 data points for AR1 estimation")
        
        if method == 'mle':
            # Maximum Likelihood Estimation
            self._fit_mle(data)
        elif method == 'ols':
            # Ordinary Least Squares
            self._fit_ols(data)
        elif method == 'yule_walker':
            # Yule-Walker estimation
            self._fit_yule_walker(data)
        else:
            raise ValueError("Method must be 'mle', 'ols', or 'yule_walker'")
        
        # Check for unit root
        if abs(self.phi) >= 0.95:
            print(f"Warning: AR1 parameter φ = {self.phi:.4f} is close to unit root")
            if not self.is_differenced and self.handle_nonstationary:
                print("Consider using differenced data or trend removal")
        
        self.fitted = True
        return self
    
    def _fit_mle(self, data):
        """
        Fit AR1 model using Maximum Likelihood Estimation.
        """
        # Use statsmodels for robust MLE estimation
        try:
            model = AutoReg(data, lags=1, trend='c')
            fitted_model = model.fit()
            self.phi = fitted_model.params[1]  # AR coefficient
            self.sigma = np.sqrt(fitted_model.sigma2)
        except:
            # Fallback to simple OLS if statsmodels fails
            self._fit_ols(data)
    
    def _fit_ols(self, data):
        """
        Fit AR1 model using Ordinary Least Squares.
        """
        # Create lagged variables
        y = data[1:]
        X = data[:-1].reshape(-1, 1)
        
        # Add constant term
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.phi = beta[1]
        
        # Calculate residuals and standard deviation
        y_pred = X @ beta
        residuals = y - y_pred
        self.sigma = np.std(residuals)
    
    def _fit_yule_walker(self, data):
        """
        Fit AR1 model using Yule-Walker equations.
        """
        # Calculate autocorrelation at lag 1
        n = len(data)
        mean_data = np.mean(data)
        centered_data = data - mean_data
        
        # Autocorrelation at lag 1
        acf1 = np.sum(centered_data[1:] * centered_data[:-1]) / (n - 1)
        acf0 = np.sum(centered_data**2) / n
        
        # Yule-Walker estimate
        self.phi = acf1 / acf0
        
        # Estimate error variance
        self.sigma = np.sqrt(acf0 * (1 - self.phi**2))
    
    def predict(self, last_value, steps=1):
        """
        Predict future values using the fitted AR1 model.
        
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
        
        predictions = []
        current_value = last_value
        
        for _ in range(steps):
            predicted = self.phi * current_value
            predictions.append(predicted)
            current_value = predicted
        
        return np.array(predictions)
    
    def predict_with_integration(self, last_value, steps=1):
        """
        Predict future values with integration (for differenced data).
        
        Parameters:
        -----------
        last_value : float
            Last observed value in the series
        steps : int
            Number of steps ahead to predict
            
        Returns:
        --------
        array
            Predicted values in original scale
        """
        if not self.fitted or not self.is_differenced:
            return self.predict(last_value, steps)
        
        # Predict in differenced space
        diff_predictions = self.predict(last_value, steps)
        
        # Integrate back to original scale
        original_predictions = []
        current_level = self.original_data[-1]  # Last original value
        
        for diff_pred in diff_predictions:
            current_level += diff_pred
            original_predictions.append(current_level)
        
        return np.array(original_predictions)
    
    def simulate(self, last_value, steps=1, n_simulations=1000):
        """
        Simulate future values using the fitted AR1 model.
        
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
        
        simulations = np.zeros((n_simulations, steps))
        
        for i in range(n_simulations):
            current_value = last_value
            
            for j in range(steps):
                # AR1 simulation: X_t = φ * X_{t-1} + ε_t
                error = np.random.normal(0, self.sigma)
                simulated = self.phi * current_value + error
                simulations[i, j] = simulated
                current_value = simulated
        
        return simulations
    
    def simulate_with_integration(self, last_value, steps=1, n_simulations=1000):
        """
        Simulate future values with integration (for differenced data).
        
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
            Simulated values in original scale
        """
        if not self.fitted or not self.is_differenced:
            return self.simulate(last_value, steps, n_simulations)
        
        # Simulate in differenced space
        diff_simulations = self.simulate(last_value, steps, n_simulations)
        
        # Integrate back to original scale
        original_simulations = np.zeros((n_simulations, steps))
        last_original = self.original_data[-1]
        
        for i in range(n_simulations):
            current_level = last_original
            for j in range(steps):
                current_level += diff_simulations[i, j]
                original_simulations[i, j] = current_level
        
        return original_simulations
    
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
            'phi': self.phi,
            'sigma': self.sigma,
            'is_differenced': self.is_differenced
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
            Residuals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating residuals")
        
        data = np.array(data)
        
        if self.is_differenced:
            # Use differenced data for residuals
            data = np.diff(data)
        
        residuals = []
        for i in range(1, len(data)):
            predicted = self.phi * data[i-1]
            residual = data[i] - predicted
            residuals.append(residual)
        
        return np.array(residuals)
    
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
        k = 2  # Number of parameters (phi, sigma)
        
        # Calculate log-likelihood
        log_likelihood = -0.5 * n * np.log(2 * np.pi * self.sigma**2) - 0.5 * np.sum(residuals**2) / self.sigma**2
        
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
        k = 2  # Number of parameters (phi, sigma)
        
        # Calculate log-likelihood
        log_likelihood = -0.5 * n * np.log(2 * np.pi * self.sigma**2) - 0.5 * np.sum(residuals**2) / self.sigma**2
        
        # Calculate BIC
        bic = k * np.log(n) - 2 * log_likelihood
        
        return bic
    
    def get_stationarity_tests(self, data=None):
        """
        Get stationarity test results.
        
        Parameters:
        -----------
        data : array-like, optional
            Data to test (uses original data if None)
            
        Returns:
        --------
        dict
            Stationarity test results
        """
        if data is None:
            data = self.original_data
        
        return self._test_stationarity(data)


def fit_ar1_model(data, method='mle', handle_nonstationary=True, force_stationary=False):
    """
    Convenience function to fit AR1 model.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    method : str
        Estimation method
    handle_nonstationary : bool
        Whether to handle non-stationary data
    force_stationary : bool
        Force stationarity by differencing
        
    Returns:
    --------
    AR1Model
        Fitted AR1 model
    """
    model = AR1Model(handle_nonstationary=handle_nonstationary)
    return model.fit(data, method=method, force_stationary=force_stationary)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data (including non-stationary example)
    n = 100
    
    # Stationary AR1 data
    phi_true = 0.7
    sigma_true = 0.1
    data_stationary = np.zeros(n)
    data_stationary[0] = 0
    
    for i in range(1, n):
        data_stationary[i] = phi_true * data_stationary[i-1] + np.random.normal(0, sigma_true)
    
    # Non-stationary data (random walk)
    data_nonstationary = np.cumsum(np.random.normal(0, 0.1, n))
    
    print("=== STATIONARY DATA EXAMPLE ===")
    model_stationary = fit_ar1_model(data_stationary, force_stationary=True)
    params = model_stationary.get_parameters()
    print(f"Fitted φ: {params['phi']:.4f}")
    print(f"True φ: {phi_true:.4f}")
    
    print("\n=== NON-STATIONARY DATA EXAMPLE ===")
    model_nonstationary = fit_ar1_model(data_nonstationary, force_stationary=True)
    params = model_nonstationary.get_parameters()
    print(f"Fitted φ: {params['phi']:.4f}")
    print(f"Is differenced: {params['is_differenced']}")
    
    # Make predictions
    prediction = model_stationary.predict(data_stationary[-1], steps=5)
    print(f"\n5-step ahead prediction: {prediction}")
    
    # Run simulations
    simulations = model_stationary.simulate(data_stationary[-1], steps=5, n_simulations=100)
    print(f"Simulation mean: {np.mean(simulations, axis=0)}")
    print(f"Simulation std: {np.std(simulations, axis=0)}")
