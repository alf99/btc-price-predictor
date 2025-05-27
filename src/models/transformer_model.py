"""
Transformer-based model for long-term BTC price prediction.
Implements a simplified version of Temporal Fusion Transformer.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, MultiHeadAttention, LayerNormalization, Dropout, 
    Input, Concatenate, GlobalAveragePooling1D, Conv1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

class TemporalFusionTransformer:
    """Temporal Fusion Transformer for long-term price prediction."""
    
    def __init__(self, 
                sequence_length: int = 30, 
                n_features: int = 20,
                n_heads: int = 4,
                hidden_dim: int = 128,
                dropout_rate: float = 0.1,
                learning_rate: float = 0.001):
        """
        Initialize Temporal Fusion Transformer model.
        
        Args:
            sequence_length: Number of time steps in each sequence
            n_features: Number of features in each time step
            n_heads: Number of attention heads
            hidden_dim: Dimension of hidden layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def _transformer_encoder(self, inputs):
        """
        Transformer encoder block.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor
        """
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=self.n_heads, key_dim=self.hidden_dim // self.n_heads
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn = Dense(self.hidden_dim * 4, activation='relu')(attention_output)
        ffn = Dense(self.hidden_dim)(ffn)
        
        # Add & Norm
        outputs = LayerNormalization(epsilon=1e-6)(attention_output + ffn)
        
        return outputs
    
    def build_model(self, output_dim: int = 1):
        """
        Build Temporal Fusion Transformer model architecture.
        
        Args:
            output_dim: Dimension of output (1 for regression, >1 for classification)
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Initial processing
        x = Conv1D(filters=self.hidden_dim, kernel_size=1, activation='relu')(inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Transformer encoder blocks
        for _ in range(2):  # Stack multiple encoder blocks
            x = self._transformer_encoder(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Final dense layers
        x = Dense(self.hidden_dim, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        if output_dim == 1:
            # Regression task
            outputs = Dense(output_dim)(x)
            loss = 'mean_squared_error'
        else:
            # Classification task
            outputs = Dense(output_dim, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=['mae'] if output_dim == 1 else ['accuracy']
        )
        
        self.model = model
        return model
    
    def build_advanced_model(self, 
                           static_features_dim: int = 0, 
                           output_dim: int = 1):
        """
        Build advanced Temporal Fusion Transformer with static features.
        
        Args:
            static_features_dim: Dimension of static features
            output_dim: Dimension of output (1 for regression, >1 for classification)
            
        Returns:
            Compiled Keras model
        """
        # Time series input
        time_series_input = Input(shape=(self.sequence_length, self.n_features), 
                                 name='time_series_input')
        
        # Static features input (if any)
        if static_features_dim > 0:
            static_input = Input(shape=(static_features_dim,), name='static_input')
            
        # Process time series with transformer
        x = Conv1D(filters=self.hidden_dim, kernel_size=1, activation='relu')(time_series_input)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Transformer encoder blocks
        for _ in range(2):
            x = self._transformer_encoder(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Combine with static features if available
        if static_features_dim > 0:
            static_features = Dense(self.hidden_dim // 2, activation='relu')(static_input)
            x = Concatenate()([x, static_features])
        
        # Final dense layers
        x = Dense(self.hidden_dim, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        if output_dim == 1:
            # Regression task
            outputs = Dense(output_dim)(x)
            loss = 'mean_squared_error'
        else:
            # Classification task
            outputs = Dense(output_dim, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        
        # Create model
        if static_features_dim > 0:
            model = Model(inputs=[time_series_input, static_input], outputs=outputs)
        else:
            model = Model(inputs=time_series_input, outputs=outputs)
        
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
    sequence_length = 30
    n_features = 20
    n_samples = 1000
    
    X = np.random.random((n_samples, sequence_length, n_features))
    y = np.random.random(n_samples)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create and train model
    model = TemporalFusionTransformer(
        sequence_length=sequence_length, 
        n_features=n_features
    )
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