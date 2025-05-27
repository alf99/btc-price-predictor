"""
Metrics utility for BTC Price Predictor.
Provides functions for evaluating model performance.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # Calculate directional accuracy
    direction_true = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'directional_accuracy': float(directional_accuracy)
    }

def calculate_profit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    initial_balance: float = 1000.0,
    transaction_fee: float = 0.001
) -> Dict[str, float]:
    """
    Calculate profit-based metrics using a simple trading strategy.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        initial_balance: Initial balance for simulation
        transaction_fee: Transaction fee as a percentage
        
    Returns:
        Dictionary of profit metrics
    """
    # Ensure arrays are 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Initialize variables
    balance = initial_balance
    btc_balance = 0.0
    trades = 0
    profitable_trades = 0
    
    # Simulate trading
    for i in range(1, len(y_true)):
        # Predict price movement
        predicted_change = y_pred[i] - y_true[i-1]
        
        # Buy BTC if price is predicted to increase
        if predicted_change > 0 and balance > 0:
            # Calculate amount of BTC to buy
            btc_to_buy = balance / y_true[i-1]
            # Apply transaction fee
            btc_to_buy *= (1 - transaction_fee)
            # Update balances
            btc_balance = btc_to_buy
            balance = 0
            trades += 1
        
        # Sell BTC if price is predicted to decrease
        elif predicted_change < 0 and btc_balance > 0:
            # Calculate amount of USD to receive
            usd_to_receive = btc_balance * y_true[i-1]
            # Apply transaction fee
            usd_to_receive *= (1 - transaction_fee)
            # Update balances
            balance = usd_to_receive
            btc_balance = 0
            trades += 1
            
            # Check if trade was profitable
            if usd_to_receive > initial_balance:
                profitable_trades += 1
    
    # Calculate final balance
    final_balance = balance + (btc_balance * y_true[-1])
    
    # Calculate profit metrics
    profit = final_balance - initial_balance
    profit_percentage = (profit / initial_balance) * 100
    
    # Calculate buy and hold profit
    buy_and_hold_profit = (y_true[-1] / y_true[0] - 1) * 100
    
    # Calculate win rate
    win_rate = (profitable_trades / trades) * 100 if trades > 0 else 0
    
    return {
        'profit': float(profit),
        'profit_percentage': float(profit_percentage),
        'buy_and_hold_profit': float(buy_and_hold_profit),
        'trades': int(trades),
        'win_rate': float(win_rate)
    }

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    include_profit_metrics: bool = True
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        include_profit_metrics: Whether to include profit metrics
        
    Returns:
        Dictionary of metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Calculate profit metrics
    if include_profit_metrics:
        profit_metrics = calculate_profit_metrics(y_test, y_pred)
        metrics.update(profit_metrics)
    
    return metrics