"""
Unit tests for LSTM model.
"""
import unittest
import numpy as np
import tensorflow as tf
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.lstm_model import LSTMModel

class TestLSTMModel(unittest.TestCase):
    """Test cases for LSTM model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_shape = (24, 10)  # 24 time steps, 10 features
        self.model = LSTMModel(
            input_shape=self.input_shape,
            lstm_units=64,
            dropout_rate=0.2,
            model_type='simple'
        )
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsInstance(self.model.model, tf.keras.Model)
    
    def test_model_output_shape(self):
        """Test model output shape."""
        # Create random input
        x = np.random.random((1, *self.input_shape))
        
        # Get model output
        y = self.model.model.predict(x)
        
        # Check output shape
        self.assertEqual(y.shape, (1, 1))
    
    def test_model_compilation(self):
        """Test model compilation."""
        self.model.compile(learning_rate=0.001)
        
        # Check if model is compiled
        self.assertIsNotNone(self.model.model.optimizer)
        self.assertIsNotNone(self.model.model.loss)
    
    def test_model_training(self):
        """Test model training."""
        # Create random data
        x = np.random.random((10, *self.input_shape))
        y = np.random.random((10, 1))
        
        # Compile model
        self.model.compile(learning_rate=0.001)
        
        # Train model
        history = self.model.model.fit(
            x, y,
            epochs=1,
            batch_size=2,
            verbose=0
        )
        
        # Check if training was successful
        self.assertIn('loss', history.history)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Create random input
        x = np.random.random((1, *self.input_shape))
        
        # Get model prediction
        y = self.model.predict(x)
        
        # Check prediction shape
        self.assertEqual(y.shape, (1, 1))
    
    def test_model_save_load(self):
        """Test model save and load."""
        # Create temporary directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Save model
        model_path = os.path.join(temp_dir, 'model')
        self.model.save(model_path)
        
        # Check if model was saved
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        loaded_model = LSTMModel.load(model_path)
        
        # Check if loaded model is instance of LSTMModel
        self.assertIsInstance(loaded_model, LSTMModel)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()