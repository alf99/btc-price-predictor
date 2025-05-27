"""
LSTM model for short-term BTC price prediction.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for short-term price prediction."""
    
    def __init__(self, 
                sequence_length: int = 24, 
                n_features: int = 10,
                lstm_units: int = 100,
                dropout_rate: float = 0.2,
                learning_rate: float = 0.001):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps in each sequence
            n_features: Number of features in each time step
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self, output_dim: int = 1):
        """
        Build LSTM model architecture.
        
        Args:
            output_dim: Dimension of output (1 for regression, >1 for classification)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(self.lstm_units, 
                      return_sequences=True, 
                      input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(self.lstm_units // 2, return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if output_dim == 1:
            # Regression task
            model.add(Dense(output_dim))
            loss = 'mean_squared_error'
        else:
            # Classification task
            model.add(Dense(output_dim, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['mae'] if output_dim == 1 else ['accuracy']
        )
        
        self.model = model
        return model
    
    def build_bidirectional_model(self, output_dim: int = 1):
        """
        Build bidirectional LSTM model architecture.
        
        Args:
            output_dim: Dimension of output (1 for regression, >1 for classification)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(self.lstm_units, return_sequences=True),
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Second Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(self.lstm_units // 2, return_sequences=False)))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if output_dim == 1:
            # Regression task
            model.add(Dense(output_dim))
            loss = 'mean_squared_error'
        else:
            # Classification task
            model.add(Dense(output_dim, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['mae'] if output_dim == 1 else ['accuracy']
        )
        
        self.model = model
        return model
    
    def build_gru_model(self, output_dim: int = 1):
        """
        Build GRU model architecture.
        
        Args:
            output_dim: Dimension of output (1 for regression, >1 for classification)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First GRU layer
        model.add(GRU(self.lstm_units, 
                     return_sequences=True, 
                     input_shape=(self.sequence_length, self.n_features)))
        model.add(Dropout(self.dropout_rate))
        
        # Second GRU layer
        model.add(GRU(self.lstm_units // 2, return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if output_dim == 1:
            # Regression task
            model.add(Dense(output_dim))
            loss = 'mean_squared_error'
        else:
            # Classification task
            model.add(Dense(output_dim, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['mae'] if output_dim == 1 else ['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
             batch_size: int = 32, 
             epochs: int = 100,
             patience: int = 10,
             model_path: str = None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            model_path: Path to save the best model
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
            )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        return self.model.predict(X)
    
    def save_model(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
        return self.model

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
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create and train model
    model = LSTMModel(sequence_length=sequence_length, n_features=n_features)
    model.build_model()
    
    print(model.model.summary())
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        epochs=5
    )
    
    # Make predictions
    predictions = model.predict(X_val[:5])
    print(f"Predictions: {predictions.flatten()}")
    print(f"Actual: {y_val[:5]}")