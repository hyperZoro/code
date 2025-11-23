"""
LSTM Model implementation for time series modeling.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class LSTMModel:
    """
    LSTM (Long Short-Term Memory) Model for time series forecasting.
    """
    
    def __init__(self, sequence_length=8, hidden_units=50, dropout_rate=0.2):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        sequence_length : int
            Number of time steps to use as input features
        hidden_units : int
            Number of LSTM units in the hidden layer
        dropout_rate : float
            Dropout rate for regularization
        """
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False
        
    def prepare_data(self, data):
        """
        Prepare data for LSTM training.
        
        Parameters:
        -----------
        data : array-like
            Time series data
            
        Returns:
        --------
        tuple
            (X, y) where X is the input sequences and y is the target values
        """
        data = np.array(data)
        
        # Normalize the data
        data_normalized = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data_normalized)):
            X.append(data_normalized[i-self.sequence_length:i])
            y.append(data_normalized[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
        """
        self.model = Sequential([
            LSTM(units=self.hidden_units, 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(units=self.hidden_units//2, 
                 return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
    def fit(self, data, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """
        Fit LSTM model to data.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
            
        Returns:
        --------
        self : LSTMModel
            Fitted model instance
        """
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Need at least {self.sequence_length + 1} data points")
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Build model
        self.build_model((X.shape[1], 1))
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.fitted = True
        self.training_history = history.history
        
        return self
    
    def predict(self, data, steps=1):
        """
        Predict future values using the fitted LSTM model.
        
        Parameters:
        -----------
        data : array-like
            Historical data for making predictions
        steps : int
            Number of steps ahead to predict
            
        Returns:
        --------
        array
            Predicted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        # Normalize the input data
        data_normalized = self.scaler.transform(np.array(data).reshape(-1, 1)).flatten()
        
        predictions = []
        current_sequence = data_normalized[-self.sequence_length:]
        
        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            pred_normalized = self.model.predict(X_pred, verbose=0)[0, 0]
            
            # Denormalize
            pred = self.scaler.inverse_transform([[pred_normalized]])[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred_normalized)
        
        return np.array(predictions)
    
    def simulate(self, data, steps=1, n_simulations=1000):
        """
        Simulate future values using the fitted LSTM model with Monte Carlo.
        
        Parameters:
        -----------
        data : array-like
            Historical data for making simulations
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
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for simulation")
        
        # Estimate residual standard deviation from training
        try:
            # Use the get_residuals method for consistency
            residuals = self.get_residuals(data)
            residual_std = np.std(residuals) if len(residuals) > 0 else 0.01
        except:
            residual_std = 0.01  # Default if estimation fails
        
        # Normalize the input data
        data_normalized = self.scaler.transform(np.array(data).reshape(-1, 1)).flatten()
        
        simulations = np.zeros((n_simulations, steps))
        
        for i in range(n_simulations):
            current_sequence = data_normalized[-self.sequence_length:].copy()
            
            for j in range(steps):
                try:
                    # Reshape for prediction
                    X_pred = current_sequence.reshape(1, self.sequence_length, 1)
                    
                    # Make prediction with noise
                    pred_normalized = self.model.predict(X_pred, verbose=0)
                    if len(pred_normalized.shape) > 1:
                        pred_normalized = pred_normalized[0, 0]
                    else:
                        pred_normalized = pred_normalized[0]
                    
                    noise = np.random.normal(0, residual_std)
                    pred_normalized += noise
                    
                    # Denormalize
                    pred = self.scaler.inverse_transform([[pred_normalized]])[0, 0]
                    simulations[i, j] = pred
                    
                    # Update sequence for next simulation
                    current_sequence = np.append(current_sequence[1:], pred_normalized)
                except Exception as e:
                    # If prediction fails, use a simple random walk
                    simulations[i, j] = simulations[i, j-1] + np.random.normal(0, 0.01) if j > 0 else data[-1] + np.random.normal(0, 0.01)
        
        return simulations
    
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
        
        try:
            # Prepare data
            data_array = np.array(data)
            if len(data_array) < self.sequence_length + 1:
                raise ValueError("Data too short for LSTM residual calculation")
            
            # Normalize data
            data_normalized = self.scaler.transform(data_array.reshape(-1, 1)).flatten()
            
            # Calculate residuals using rolling window approach
            residuals = []
            for i in range(self.sequence_length, len(data_normalized)):
                # Get input sequence
                input_seq = data_normalized[i-self.sequence_length:i]
                X_input = input_seq.reshape(1, self.sequence_length, 1)
                
                # Make prediction
                pred_normalized = self.model.predict(X_input, verbose=0)
                
                # Handle different output shapes
                if len(pred_normalized.shape) > 1:
                    pred_normalized = pred_normalized[0, 0]
                else:
                    pred_normalized = pred_normalized[0]
                
                # Calculate residual in normalized space
                actual_normalized = data_normalized[i]
                residual_normalized = actual_normalized - pred_normalized
                
                # Denormalize residual
                residual_denorm = self.scaler.inverse_transform([[residual_normalized]])[0, 0]
                residuals.append(residual_denorm)
            
            return np.array(residuals)
            
        except Exception as e:
            # If there's an error, return simple residuals based on differences
            print(f"Warning: Could not calculate LSTM residuals properly: {e}")
            # Return simple first differences as residuals
            return np.diff(data[self.sequence_length:])
    
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
        k = self.model.count_params()  # Number of model parameters
        
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
        k = self.model.count_params()  # Number of model parameters
        
        # Calculate log-likelihood (assuming normal residuals)
        residual_std = np.std(residuals)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * residual_std**2) - 0.5 * np.sum(residuals**2) / residual_std**2
        
        # Calculate BIC
        bic = k * np.log(n) - 2 * log_likelihood
        
        return bic
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.fitted:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No fitted model to save")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.fitted = True
        print(f"Model loaded from {filepath}")


def fit_lstm_model(data, sequence_length=8, hidden_units=50, dropout_rate=0.2, 
                   validation_split=0.2, epochs=100, batch_size=32, verbose=1):
    """
    Convenience function to fit LSTM model.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    sequence_length : int
        Number of time steps to use as input features
    hidden_units : int
        Number of LSTM units in the hidden layer
    dropout_rate : float
        Dropout rate for regularization
    validation_split : float
        Fraction of data to use for validation
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    verbose : int
        Verbosity level
        
    Returns:
    --------
    LSTMModel
        Fitted LSTM model
    """
    model = LSTMModel(sequence_length=sequence_length, 
                     hidden_units=hidden_units, 
                     dropout_rate=dropout_rate)
    return model.fit(data, validation_split=validation_split, 
                    epochs=epochs, batch_size=batch_size, verbose=verbose)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample time series data
    n = 200
    t = np.arange(n)
    data = 0.1 * t + 0.8 * np.sin(0.1 * t) + np.random.normal(0, 0.1, n)
    
    # Fit LSTM model
    model = fit_lstm_model(data, sequence_length=8, epochs=50, verbose=0)
    
    # Make prediction
    prediction = model.predict(data, steps=5)
    print(f"5-step ahead prediction: {prediction}")
    
    # Simulate
    simulations = model.simulate(data, steps=5, n_simulations=100)
    print(f"Simulation mean: {np.mean(simulations, axis=0)}")
    print(f"Simulation std: {np.std(simulations, axis=0)}")
    
    # Calculate metrics
    aic = model.get_aic(data)
    bic = model.get_bic(data)
    print(f"AIC: {aic:.2f}")
    print(f"BIC: {bic:.2f}")
