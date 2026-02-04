"""
Visualization utilities for FX rate modeling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Visualization tools for FX rate analysis and model evaluation.
    """
    
    def __init__(self, output_dir: str = "./plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_fx_rates(
        self,
        df: pd.DataFrame,
        title: str = "FX Rates",
        save_path: Optional[str] = None
    ):
        """
        Plot FX rates over time.
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in df.columns:
            ax.plot(df.index, df[col], label=col, linewidth=1.5)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Exchange Rate")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_returns(
        self,
        df: pd.DataFrame,
        title: str = "FX Returns",
        save_path: Optional[str] = None
    ):
        """
        Plot returns distribution.
        """
        returns = df.pct_change().dropna()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time series of returns
        for col in returns.columns:
            axes[0].plot(returns.index, returns[col], label=col, alpha=0.7)
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Return")
        axes[0].set_title("Returns Over Time")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Distribution
        returns_melted = returns.melt(var_name='FX Pair', value_name='Return')
        sns.histplot(data=returns_melted, x='Return', hue='FX Pair', kde=True, ax=axes[1])
        axes[1].set_title("Returns Distribution")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None
    ):
        """
        Plot actual vs predicted values.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(y_true), freq='D')
        
        # Time series plot
        axes[0].plot(dates, y_true, label='Actual', linewidth=1.5, color='blue')
        axes[0].plot(dates, y_pred, label='Predicted', linewidth=1.5, color='red', alpha=0.8)
        axes[0].fill_between(dates, y_true, y_pred, alpha=0.2, color='gray')
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Value")
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        axes[1].set_xlabel("Actual")
        axes[1].set_ylabel("Predicted")
        axes[1].set_title("Actual vs Predicted Scatter")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Model Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of multiple models.
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        if dates is None:
            dates = pd.date_range(start='2020-01-01', periods=len(y_true), freq='D')
        
        # Plot actual
        ax.plot(dates, y_true, label='Actual', linewidth=2, color='black', linestyle='-')
        
        # Plot predictions
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (name, y_pred), color in zip(predictions.items(), colors):
            ax.plot(dates, y_pred, label=name, linewidth=1.5, alpha=0.8, color=color)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(
        self,
        results_df: pd.DataFrame,
        title: str = "Metrics Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing metrics across models.
        """
        metrics = [col for col in results_df.columns if col != 'horizon']
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2,
            figsize=(14, 4 * ((n_metrics + 1) // 2))
        )
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            results_df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residuals Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot residuals analysis.
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals, linewidth=1, color='blue')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title("Residuals Over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Residual")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].set_title("Residuals Distribution")
        axes[0, 1].set_xlabel("Residual")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normality)")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Predicted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title("Residuals vs Predicted")
        axes[1, 1].set_xlabel("Predicted")
        axes[1, 1].set_ylabel("Residual")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        title: str = "Correlation Matrix",
        save_path: Optional[str] = None
    ):
        """
        Plot correlation matrix heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_volatility(
        self,
        df: pd.DataFrame,
        window: int = 20,
        title: str = "Rolling Volatility",
        save_path: Optional[str] = None
    ):
        """
        Plot rolling volatility.
        """
        returns = df.pct_change().dropna()
        volatility = returns.rolling(window).std() * np.sqrt(252)  # Annualized
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in volatility.columns:
            ax.plot(volatility.index, volatility[col], label=col, linewidth=1.5)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Annualized Volatility")
        ax.set_title(f"{title} ({window}-day window)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[str] = None
    ):
        """
        Plot training loss history.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label='Train Loss', linewidth=1.5)
        
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss', linewidth=1.5)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.output_dir, save_path), dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test visualization
    np.random.seed(42)
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    df = pd.DataFrame({
        'EURUSD': np.cumsum(np.random.randn(252) * 0.01) + 1.10,
        'GBPUSD': np.cumsum(np.random.randn(252) * 0.01) + 1.35,
        'USDJPY': np.cumsum(np.random.randn(252) * 0.01) + 110.0
    }, index=dates)
    
    viz = Visualizer()
    
    viz.plot_fx_rates(df, "FX Rates Test", "test_fx_rates.png")
    viz.plot_returns(df, "Returns Test", "test_returns.png")
    viz.plot_correlation_matrix(df, "Correlation Test", "test_correlation.png")
    viz.plot_volatility(df, window=20, title="Volatility Test", save_path="test_volatility.png")
    
    print("Test plots saved to ./plots/")