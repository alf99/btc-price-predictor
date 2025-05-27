"""
Visualization utility for BTC Price Predictor.
Provides functions for visualizing data and model predictions.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import io
import base64

def plot_price_history(
    df: pd.DataFrame,
    price_col: str = 'close',
    date_col: str = 'open_time',
    title: str = 'Bitcoin Price History',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot price history.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        date_col: Column name for date
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[date_col], df[price_col])
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'close',
    date_col: str = 'open_time',
    sma_cols: List[str] = ['sma_7', 'sma_30'],
    rsi_col: str = 'rsi_14',
    macd_cols: List[str] = ['macd', 'macd_signal', 'macd_hist'],
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Plot technical indicators.
    
    Args:
        df: DataFrame with price and indicator data
        price_col: Column name for price
        date_col: Column name for date
        sma_cols: Column names for SMA indicators
        rsi_col: Column name for RSI indicator
        macd_cols: Column names for MACD indicators
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot price and moving averages
    axes[0].plot(df[date_col], df[price_col], label='Price')
    for sma in sma_cols:
        if sma in df.columns:
            axes[0].plot(df[date_col], df[sma], label=sma.upper())
    axes[0].set_title('Price and Moving Averages')
    axes[0].set_ylabel('Price (USD)')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot RSI
    if rsi_col in df.columns:
        axes[1].plot(df[date_col], df[rsi_col])
        axes[1].axhline(y=70, color='r', linestyle='-')
        axes[1].axhline(y=30, color='g', linestyle='-')
        axes[1].set_title('RSI')
        axes[1].set_ylabel('RSI')
        axes[1].grid(True)
    
    # Plot MACD
    if all(col in df.columns for col in macd_cols[:2]):
        axes[2].plot(df[date_col], df[macd_cols[0]], label='MACD')
        axes[2].plot(df[date_col], df[macd_cols[1]], label='Signal')
        if macd_cols[2] in df.columns:
            axes[2].bar(df[date_col], df[macd_cols[2]], alpha=0.3, label='Histogram')
        axes[2].set_title('MACD')
        axes[2].set_ylabel('MACD')
        axes[2].set_xlabel('Date')
        axes[2].grid(True)
        axes[2].legend()
    
    plt.tight_layout()
    return fig

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = 'Actual vs Predicted Bitcoin Prices',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot actual vs predicted prices.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        dates: Dates for x-axis
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        ax.plot(dates, y_true, label='Actual')
        ax.plot(dates, y_pred, label='Predicted')
        ax.set_xlabel('Date')
    else:
        ax.plot(y_true, label='Actual')
        ax.plot(y_pred, label='Predicted')
        ax.set_xlabel('Time')
    
    ax.set_title(title)
    ax.set_ylabel('Price (USD)')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_model_metrics(
    metrics: Dict[str, float],
    title: str = 'Model Performance Metrics',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot model performance metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Select metrics to plot
    plot_metrics = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mape': 'MAPE (%)',
        'directional_accuracy': 'Directional Accuracy (%)'
    }
    
    # Filter metrics
    metrics_to_plot = {k: metrics[k] for k in plot_metrics if k in metrics}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(
        range(len(metrics_to_plot)),
        list(metrics_to_plot.values()),
        tick_label=[plot_metrics[k] for k in metrics_to_plot]
    )
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )
    
    ax.set_title(title)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    return fig

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert Matplotlib figure to base64 string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str