"""
Model utility for BTC Price Predictor.
Provides functions for saving and loading models.
"""
import os
import json
import pickle
import logging
import tensorflow as tf
from typing import Dict, Any, Optional, Union, Tuple
import datetime

logger = logging.getLogger(__name__)

def save_model(
    model: Union[tf.keras.Model, Any],
    model_name: str,
    model_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    models_dir: str = 'models'
) -> str:
    """
    Save model and metadata.
    
    Args:
        model: Model to save
        model_name: Name of the model
        model_type: Type of model ('lstm', 'transformer', etc.)
        metadata: Additional metadata
        models_dir: Directory to save models
        
    Returns:
        Path to saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(models_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    if isinstance(model, tf.keras.Model):
        model_path = os.path.join(model_dir, 'model')
        model.save(model_path)
        logger.info(f"Saved TensorFlow model to {model_path}")
    else:
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved pickle model to {model_path}")
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'tensorflow_model': isinstance(model, tf.keras.Model)
    })
    
    # Save metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved model metadata to {metadata_path}")
    
    return model_dir

def load_model(
    model_dir: str
) -> Tuple[Union[tf.keras.Model, Any], Dict[str, Any]]:
    """
    Load model and metadata.
    
    Args:
        model_dir: Directory containing model and metadata
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    if metadata.get('tensorflow_model', False):
        model_path = os.path.join(model_dir, 'model')
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded TensorFlow model from {model_path}")
    else:
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded pickle model from {model_path}")
    
    return model, metadata

def list_models(
    models_dir: str = 'models'
) -> Dict[str, Dict[str, Any]]:
    """
    List available models.
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        Dictionary of model names and metadata
    """
    models = {}
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory {models_dir} does not exist")
        return models
    
    # List model directories
    for model_dir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_dir)
        
        # Check if directory
        if not os.path.isdir(model_path):
            continue
        
        # Check if metadata exists
        metadata_path = os.path.join(model_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            continue
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Add to models dictionary
        models[model_dir] = metadata
    
    return models

def get_latest_model(
    model_type: Optional[str] = None,
    models_dir: str = 'models'
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Get latest model of specified type.
    
    Args:
        model_type: Type of model ('lstm', 'transformer', etc.)
        models_dir: Directory containing models
        
    Returns:
        Tuple of (model_dir, metadata) or None if no models found
    """
    models = list_models(models_dir)
    
    if not models:
        return None
    
    # Filter by model type
    if model_type:
        models = {k: v for k, v in models.items() if v.get('model_type') == model_type}
    
    if not models:
        return None
    
    # Sort by timestamp
    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1].get('timestamp', ''),
        reverse=True
    )
    
    # Return latest model
    model_dir = sorted_models[0][0]
    metadata = sorted_models[0][1]
    
    return os.path.join(models_dir, model_dir), metadata