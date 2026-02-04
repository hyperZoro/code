"""
PyTorch-based deep learning models for FX rate forecasting.
Includes LSTM, GRU, and Transformer models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import os


# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class TimeSeriesDataset(Dataset):
    """Dataset for time series sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BasePyTorchModel(nn.Module):
    """Base class for PyTorch time series models."""
    
    def __init__(self, name: str, config: Dict):
        super().__init__()
        self.name = name
        self.config = config
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        """
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 32))
        
        # Optimizer and scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.config.get('learning_rate', 0.001))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
        
        # Training loop
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('early_stopping_patience', 10)
        
        self.to(DEVICE)
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            avg_val_loss = None
            if val_loader is not None:
                self.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        outputs = self(X_batch)
                        loss = criterion(outputs.squeeze(), y_batch)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                self.history['val_loss'].append(avg_val_loss)
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}"
                if avg_val_loss is not None:
                    msg += f" - Val Loss: {avg_val_loss:.6f}"
                print(msg)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        self.eval()
        self.to(DEVICE)
        
        dataset = TimeSeriesDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=self.config.get('batch_size', 32))
        
        predictions = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(DEVICE)
                outputs = self(X_batch)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        return np.array(predictions)


class LSTMModel(BasePyTorchModel):
    """
    LSTM (Long Short-Term Memory) model for time series forecasting.
    """
    
    def __init__(self, input_size: int, config: Dict):
        super().__init__("LSTM", config)
        
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Use last hidden state
        out = self.fc(lstm_out[:, -1, :])
        return out


class GRUModel(BasePyTorchModel):
    """
    GRU (Gated Recurrent Unit) model for time series forecasting.
    """
    
    def __init__(self, input_size: int, config: Dict):
        super().__init__("GRU", config)
        
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        gru_out, hidden = self.gru(x)
        # Use last hidden state
        out = self.fc(gru_out[:, -1, :])
        return out


class TransformerModel(BasePyTorchModel):
    """
    Transformer model for time series forecasting.
    """
    
    def __init__(self, input_size: int, config: Dict):
        super().__init__("Transformer", config)
        
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 4)
        self.num_encoder_layers = config.get('num_encoder_layers', 2)
        self.dim_feedforward = config.get('dim_feedforward', 256)
        self.dropout = config.get('dropout', 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=self.num_encoder_layers
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use mean of all time steps
        x = x.mean(dim=1)
        out = self.fc(x)
        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class PyTorchModelWrapper:
    """
    Wrapper to handle PyTorch models with sklearn-like interface.
    """
    
    def __init__(self, model_class, input_size: int, config: Dict):
        self.model_class = model_class
        self.input_size = input_size
        self.config = config
        self.model = None
        self.name = model_class.__name__
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """Fit the model."""
        self.model = self.model_class(self.input_size, self.config)
        self.model.fit(X_train, y_train, X_val, y_val, verbose)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
    
    def save(self, path: str):
        """Save model."""
        if self.model is not None:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self.model.save_checkpoint(path)
    
    def load(self, path: str):
        """Load model."""
        self.model = self.model_class(self.input_size, self.config)
        self.model.load_checkpoint(path)


if __name__ == "__main__":
    # Test PyTorch models
    print("Testing LSTM Model")
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    seq_len = 60
    n_features = 5
    
    X = np.random.randn(n_samples, seq_len, n_features)
    y = np.random.randn(n_samples)
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Test LSTM
    config = {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 5,
        'early_stopping_patience': 5
    }
    
    model = PyTorchModelWrapper(LSTMModel, n_features, config)
    model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_val)
    
    print(f"Predictions shape: {predictions.shape}")