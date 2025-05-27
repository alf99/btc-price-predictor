"""
Configuration utility for BTC Price Predictor.
Loads and provides access to application configuration.
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or os.path.join('config', 'config.json')
        self.config = self._load_config()
    
    def _load_config(self):
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return self._default_config()
    
    def _default_config(self):
        """
        Return default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "websocket_port": 8765
            },
            "data": {
                "binance": {
                    "symbol": "BTCUSDT",
                    "intervals": ["1h", "4h", "1d"],
                    "default_interval": "1h",
                    "limit": 1000
                },
                "coingecko": {
                    "coin_id": "bitcoin",
                    "vs_currency": "usd",
                    "days": 30
                },
                "update_interval": 300
            },
            "models": {
                "lstm": {
                    "sequence_length": 24,
                    "lstm_units": 100,
                    "dropout_rate": 0.2,
                    "learning_rate": 0.001
                },
                "transformer": {
                    "sequence_length": 30,
                    "n_heads": 4,
                    "hidden_dim": 128,
                    "dropout_rate": 0.1,
                    "learning_rate": 0.001
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key, default=None):
        """
        Get configuration value.
        
        Args:
            key: Configuration key (can be nested with dots, e.g. 'api.port')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key, value):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (can be nested with dots, e.g. 'api.port')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

# Create global configuration instance
config = Config()