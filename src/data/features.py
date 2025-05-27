"""
Feature engineering module for BTC Price Predictor.
Transforms raw data into features suitable for machine learning models.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Handles feature engineering for cryptocurrency price data."""
    
    def __init__(self):
        pass
        
    def add_technical_indicators(self, df: pd.DataFrame, 
                                price_col: str = 'close',
                                volume_col: Optional[str] = 'volume') -> pd.DataFrame:
        """
        Add common technical indicators to the dataframe.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price data
            volume_col: Column name for volume data (if available)
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Simple Moving Averages
        for window in [7, 14, 30, 50, 200]:
            result[f'sma_{window}'] = result[price_col].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in [7, 14, 30, 50, 200]:
            result[f'ema_{window}'] = result[price_col].ewm(span=window, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = result[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.finfo(float).eps)
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window in [20]:
            mid_band = result[price_col].rolling(window=window).mean()
            std_dev = result[price_col].rolling(window=window).std()
            result[f'bb_upper_{window}'] = mid_band + (std_dev * 2)
            result[f'bb_middle_{window}'] = mid_band
            result[f'bb_lower_{window}'] = mid_band - (std_dev * 2)
            
            # Bollinger Band Width
            result[f'bb_width_{window}'] = (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}']) / result[f'bb_middle_{window}']
        
        # MACD
        ema_12 = result[price_col].ewm(span=12, adjust=False).mean()
        ema_26 = result[price_col].ewm(span=26, adjust=False).mean()
        result['macd'] = ema_12 - ema_26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # Add volume-based indicators if volume data is available
        if volume_col and volume_col in df.columns:
            # Volume Moving Average
            result['volume_sma_20'] = result[volume_col].rolling(window=20).mean()
            
            # On-Balance Volume (OBV)
            result['obv'] = np.where(
                result[price_col] > result[price_col].shift(1),
                result[volume_col],
                np.where(
                    result[price_col] < result[price_col].shift(1),
                    -result[volume_col],
                    0
                )
            ).cumsum()
        
        # Price Rate of Change
        for window in [1, 7, 14]:
            result[f'price_roc_{window}'] = result[price_col].pct_change(periods=window) * 100
        
        # Log Returns
        result['log_return'] = np.log(result[price_col] / result[price_col].shift(1))
        
        return result
    
    def create_time_features(self, df: pd.DataFrame, 
                           datetime_col: str = 'open_time') -> pd.DataFrame:
        """
        Extract time-based features from datetime column.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of the datetime column
            
        Returns:
            DataFrame with added time features
        """
        result = df.copy()
        
        # Ensure datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(result[datetime_col]):
            result[datetime_col] = pd.to_datetime(result[datetime_col])
        
        # Extract time components
        result['hour'] = result[datetime_col].dt.hour
        result['day'] = result[datetime_col].dt.day
        result['day_of_week'] = result[datetime_col].dt.dayofweek
        result['month'] = result[datetime_col].dt.month
        result['year'] = result[datetime_col].dt.year
        result['quarter'] = result[datetime_col].dt.quarter
        
        # Create cyclical features for time components
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['day_sin'] = np.sin(2 * np.pi * result['day'] / 31)
        result['day_cos'] = np.cos(2 * np.pi * result['day'] / 31)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # Is weekend
        result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
        
        return result
    
    def create_target_variables(self, df: pd.DataFrame, 
                              price_col: str = 'close',
                              horizons: List[int] = [1, 6, 24, 72]) -> pd.DataFrame:
        """
        Create target variables for different prediction horizons.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price data
            horizons: List of time steps ahead to predict
            
        Returns:
            DataFrame with added target variables
        """
        result = df.copy()
        
        # Future price and returns for different horizons
        for horizon in horizons:
            # Future price
            result[f'future_price_{horizon}'] = result[price_col].shift(-horizon)
            
            # Price change
            result[f'price_change_{horizon}'] = result[f'future_price_{horizon}'] - result[price_col]
            
            # Percentage change
            result[f'pct_change_{horizon}'] = (result[f'future_price_{horizon}'] / result[price_col] - 1) * 100
            
            # Direction (binary classification target)
            result[f'direction_{horizon}'] = (result[f'future_price_{horizon}'] > result[price_col]).astype(int)
        
        return result
    
    def prepare_sequence_data(self, df: pd.DataFrame, 
                            feature_cols: List[str],
                            target_col: str,
                            sequence_length: int = 24) -> tuple:
        """
        Prepare sequence data for time series models like LSTM.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            sequence_length: Number of time steps in each sequence
            
        Returns:
            Tuple of (X, y) where X is a 3D array of sequences and y is the target
        """
        # Drop rows with NaN values
        data = df.dropna().reset_index(drop=True)
        
        # Extract features and target
        features = data[feature_cols].values
        target = data[target_col].values
        
        X, y = [], []
        
        # Create sequences
        for i in range(len(data) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def normalize_features(self, train_df: pd.DataFrame, 
                         test_df: pd.DataFrame,
                         feature_cols: List[str]) -> tuple:
        """
        Normalize features using training data statistics.
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            feature_cols: List of feature column names to normalize
            
        Returns:
            Tuple of (normalized_train_df, normalized_test_df, scaler)
        """
        from sklearn.preprocessing import StandardScaler
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit scaler on training data
        scaler.fit(train_df[feature_cols])
        
        # Transform both training and testing data
        train_normalized = train_df.copy()
        test_normalized = test_df.copy()
        
        train_normalized[feature_cols] = scaler.transform(train_df[feature_cols])
        test_normalized[feature_cols] = scaler.transform(test_df[feature_cols])
        
        return train_normalized, test_normalized, scaler

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    data = {
        'open_time': dates,
        'open': np.random.normal(20000, 1000, 100),
        'high': np.random.normal(20500, 1000, 100),
        'low': np.random.normal(19500, 1000, 100),
        'close': np.random.normal(20200, 1000, 100),
        'volume': np.random.normal(100, 20, 100)
    }
    df = pd.DataFrame(data)
    
    # Create feature engineer
    fe = FeatureEngineer()
    
    # Add technical indicators
    df_with_indicators = fe.add_technical_indicators(df)
    print(f"Added technical indicators. Shape: {df_with_indicators.shape}")
    
    # Add time features
    df_with_time = fe.create_time_features(df_with_indicators)
    print(f"Added time features. Shape: {df_with_time.shape}")
    
    # Create target variables
    df_with_targets = fe.create_target_variables(df_with_time)
    print(f"Added target variables. Shape: {df_with_targets.shape}")