"""
Model trainer module for BTC Price Predictor.
Handles the training pipeline for different models.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import joblib
from typing import List, Dict, Union, Optional, Tuple
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TemporalFusionTransformer

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training and evaluation of models."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def prepare_data(self, df: pd.DataFrame, 
                   feature_cols: List[str],
                   target_col: str,
                   sequence_length: int,
                   test_size: float = 0.2,
                   validation_size: float = 0.1) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            sequence_length: Number of time steps in each sequence
            test_size: Proportion of data to use for testing
            validation_size: Proportion of training data to use for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Drop rows with NaN values
        data = df.dropna().reset_index(drop=True)
        
        # Extract features and target
        features = data[feature_cols].values
        target = data[target_col].values
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train+val and test sets (time-ordered)
        split_idx = int(len(X) * (1 - test_size))
        X_train_val, X_test = X[:split_idx], X[split_idx:]
        y_train_val, y_test = y[:split_idx], y[split_idx:]
        
        # Split train+val into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=validation_size,
            shuffle=False  # Keep time order
        )
        
        logger.info(f"Data shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_lstm_model(self, 
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       model_name: str = 'lstm_model',
                       **kwargs) -> LSTMModel:
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name for the model
            **kwargs: Additional arguments for model initialization and training
            
        Returns:
            Trained LSTM model
        """
        # Get model parameters
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        # Get model hyperparameters from kwargs or use defaults
        lstm_units = kwargs.get('lstm_units', 100)
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        learning_rate = kwargs.get('learning_rate', 0.001)
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        patience = kwargs.get('patience', 10)
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join('models', model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model
        model = LSTMModel(
            sequence_length=sequence_length,
            n_features=n_features,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Build model architecture
        model_type = kwargs.get('model_type', 'lstm')
        if model_type == 'bidirectional':
            model.build_bidirectional_model()
        elif model_type == 'gru':
            model.build_gru_model()
        else:
            model.build_model()
        
        # Set up TensorBoard logging
        log_dir = os.path.join('logs', model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # Train model
        model_path = os.path.join(model_dir, 'best_model.h5')
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            model_path=model_path
        )
        
        # Save model
        model.save_model(os.path.join(model_dir, 'final_model.h5'))
        
        # Store model in dictionary
        self.models[model_name] = model
        
        return model
    
    def train_transformer_model(self, 
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_val: np.ndarray,
                              y_val: np.ndarray,
                              model_name: str = 'transformer_model',
                              **kwargs) -> TemporalFusionTransformer:
        """
        Train Transformer model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name for the model
            **kwargs: Additional arguments for model initialization and training
            
        Returns:
            Trained Transformer model
        """
        # Get model parameters
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        # Get model hyperparameters from kwargs or use defaults
        n_heads = kwargs.get('n_heads', 4)
        hidden_dim = kwargs.get('hidden_dim', 128)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        learning_rate = kwargs.get('learning_rate', 0.001)
        batch_size = kwargs.get('batch_size', 32)
        epochs = kwargs.get('epochs', 100)
        patience = kwargs.get('patience', 10)
        
        # Create model directory if it doesn't exist
        model_dir = os.path.join('models', model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model
        model = TemporalFusionTransformer(
            sequence_length=sequence_length,
            n_features=n_features,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        # Build model architecture
        model_type = kwargs.get('model_type', 'basic')
        if model_type == 'advanced':
            static_features_dim = kwargs.get('static_features_dim', 0)
            model.build_advanced_model(static_features_dim=static_features_dim)
        else:
            model.build_model()
        
        # Set up TensorBoard logging
        log_dir = os.path.join('logs', model_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # Train model
        model_path = os.path.join(model_dir, 'best_model.h5')
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            model_path=model_path
        )
        
        # Save model
        model.save_model(os.path.join(model_dir, 'final_model.h5'))
        
        # Store model in dictionary
        self.models[model_name] = model
        
        return model
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test).flatten()
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        direction_actual = np.diff(y_test) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_actual == direction_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        
        return metrics
    
    def hyperparameter_tuning(self, 
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            model_type: str = 'lstm',
                            n_trials: int = 20) -> Dict:
        """
        Perform hyperparameter tuning using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Type of model ('lstm' or 'transformer')
            n_trials: Number of trials for hyperparameter search
            
        Returns:
            Dictionary with best hyperparameters
        """
        import optuna
        
        def objective(trial):
            # Common hyperparameters
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            
            if model_type == 'lstm':
                # LSTM-specific hyperparameters
                lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128, 256])
                model_subtype = trial.suggest_categorical('model_type', ['lstm', 'bidirectional', 'gru'])
                
                # Create and train model
                model = LSTMModel(
                    sequence_length=X_train.shape[1],
                    n_features=X_train.shape[2],
                    lstm_units=lstm_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
                
                if model_subtype == 'bidirectional':
                    model.build_bidirectional_model()
                elif model_subtype == 'gru':
                    model.build_gru_model()
                else:
                    model.build_model()
                
            elif model_type == 'transformer':
                # Transformer-specific hyperparameters
                n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
                hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
                
                # Create and train model
                model = TemporalFusionTransformer(
                    sequence_length=X_train.shape[1],
                    n_features=X_train.shape[2],
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
                
                model.build_model()
            
            # Train model with early stopping
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                batch_size=batch_size,
                epochs=50,  # Limit epochs for tuning
                patience=5
            )
            
            # Return best validation loss
            return min(history.history['val_loss'])
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation loss: {study.best_value}")
        
        return study.best_params

# Example usage
if __name__ == "__main__":
    # Create sample data
    sequence_length = 24
    n_features = 10
    n_samples = 1000
    
    X = np.random.random((n_samples, sequence_length, n_features))
    y = np.random.random(n_samples)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Further split training data into train and validation
    split2 = int(0.8 * len(X_train))
    X_train, X_val = X_train[:split2], X_train[split2:]
    y_train, y_val = y_train[:split2], y_train[split2:]
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Train LSTM model
    lstm_model = trainer.train_lstm_model(
        X_train, y_train,
        X_val, y_val,
        model_name='lstm_test',
        epochs=5
    )
    
    # Evaluate model
    metrics = trainer.evaluate_model(lstm_model, X_test, y_test)
    print(f"LSTM model metrics: {metrics}")