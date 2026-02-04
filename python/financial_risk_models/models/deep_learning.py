import torch
import torch.nn as nn
from torch.utils.data import Dataset

class FinancialTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (np.array): Input sequences (N, seq_len, features)
            y (np.array): Targets (N, ) or (N, output_dim)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ensure y is at least 2D (N, 1) if it's currently 1D (N,)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BaseDLModel(nn.Module):
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class LSTMModel(BaseDLModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq, feature)
        # LSTM output: (batch, seq, hidden)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class TransformerModel(BaseDLModel):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x: (batch, seq, feature)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Take last time step
        x = self.fc(x[:, -1, :])
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq, feature)
        # pe needs to be sliced to seq_len
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

import numpy as np

class TCNModel(BaseDLModel):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        """
        Simple TCN implementation.
        num_channels: list of integers, e.g. [32, 32, 32] defines depth and width.
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq, feature)
        # TCN expects (batch, channels, seq)
        x = x.transpose(1, 2)
        y = self.network(x) # (batch, channels, out_seq)
        # We want prediction based on the entire history up to t, effectively standard causal conv
        # The padding scheme above (causal) results in output length >= input length.
        # We take the last element.
        return self.fc(y[:, :, -1])

