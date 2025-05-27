"""
Preprocessing utility for BTC Price Predictor.
Provides functions for preprocessing data for model training.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Dict, Any, Optional, Union

def create_sequences(
    data: np.ndarray,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series data.
    
    Args:
        data: Input data (features and target)
        sequence_length: Length of sequences
        
    Returns:
        Tuple of (X, y) where X is the sequences and y is the targets
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length, -1])  # Last column is the target
    
    return np.array(X), np.array(y)

def create_sequences_multi_target(
    data: np.ndarray,
    sequence_length: int,
    n_targets: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series data with multiple targets.
    
    Args:
        data: Input data (features and targets)
        sequence_length: Length of sequences
        n_targets: Number of target variables
        
    Returns:
        Tuple of (X, y) where X is the sequences and y is the targets
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-n_targets])
        y.append(data[i + sequence_length, -n_targets:])
    
    return np.array(X), np.array(y)

def train_val_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for test set
        validation_size: Proportion of data for validation set
        
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    # Calculate split indices
    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - validation_size))
    
    # Split data
    train = df.iloc[:val_idx].copy()
    val = df.iloc[val_idx:test_idx].copy()
    test = df.iloc[test_idx:].copy()
    
    return train, val, test

def scale_data(
    train: np.ndarray,
    val: Optional[np.ndarray] = None,
    test: Optional[np.ndarray] = None,
    scaler_type: str = 'minmax'
) -> Tuple[np.ndarray, ...]:
    """
    Scale data using MinMaxScaler or StandardScaler.
    
    Args:
        train: Training data
        val: Validation data
        test: Test data
        scaler_type: Type of scaler ('minmax' or 'standard')
        
    Returns:
        Tuple of scaled data and scaler
    """
    # Create scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit scaler on training data
    scaler.fit(train)
    
    # Transform data
    train_scaled = scaler.transform(train)
    
    result = [train_scaled, scaler]
    
    if val is not None:
        val_scaled = scaler.transform(val)
        result.insert(1, val_scaled)
    
    if test is not None:
        test_scaled = scaler.transform(test)
        result.insert(-1, test_scaled)
    
    return tuple(result)

def prepare_data_for_lstm(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    scaler_type: str = 'minmax'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Prepare data for LSTM model.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        sequence_length: Length of sequences
        test_size: Proportion of data for test set
        validation_size: Proportion of data for validation set
        scaler_type: Type of scaler ('minmax' or 'standard')
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    # Select features and target
    data = df[feature_cols + [target_col]].copy()
    
    # Split data
    train_df, val_df, test_df = train_val_test_split(
        data,
        test_size=test_size,
        validation_size=validation_size
    )
    
    # Scale data
    train_data, val_data, test_data, scaler = scale_data(
        train_df.values,
        val_df.values,
        test_df.values,
        scaler_type=scaler_type
    )
    
    # Create sequences
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_val, y_val = create_sequences(val_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def prepare_data_for_transformer(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    sequence_length: int,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    scaler_type: str = 'minmax'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Prepare data for Transformer model.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        target_cols: List of target column names
        sequence_length: Length of sequences
        test_size: Proportion of data for test set
        validation_size: Proportion of data for validation set
        scaler_type: Type of scaler ('minmax' or 'standard')
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scalers)
    """
    # Select features and targets
    data = df[feature_cols + target_cols].copy()
    
    # Split data
    train_df, val_df, test_df = train_val_test_split(
        data,
        test_size=test_size,
        validation_size=validation_size
    )
    
    # Scale features
    feature_scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    feature_scaler.fit(train_df[feature_cols].values)
    
    train_features = feature_scaler.transform(train_df[feature_cols].values)
    val_features = feature_scaler.transform(val_df[feature_cols].values)
    test_features = feature_scaler.transform(test_df[feature_cols].values)
    
    # Scale targets
    target_scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    target_scaler.fit(train_df[target_cols].values)
    
    train_targets = target_scaler.transform(train_df[target_cols].values)
    val_targets = target_scaler.transform(val_df[target_cols].values)
    test_targets = target_scaler.transform(test_df[target_cols].values)
    
    # Create sequences for features
    X_train, y_train = [], []
    for i in range(len(train_features) - sequence_length):
        X_train.append(train_features[i:i + sequence_length])
        y_train.append(train_targets[i + sequence_length])
    
    X_val, y_val = [], []
    for i in range(len(val_features) - sequence_length):
        X_val.append(val_features[i:i + sequence_length])
        y_val.append(val_targets[i + sequence_length])
    
    X_test, y_test = [], []
    for i in range(len(test_features) - sequence_length):
        X_test.append(test_features[i:i + sequence_length])
        y_test.append(test_targets[i + sequence_length])
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Create scalers dictionary
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scalers