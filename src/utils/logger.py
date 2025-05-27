"""
Logging utility for BTC Price Predictor.
Configures and provides access to application logging.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
import sys

from src.utils.config import config

def setup_logging():
    """
    Set up logging for the application.
    
    Returns:
        Root logger
    """
    # Get logging configuration
    log_level = getattr(logging, config.get('logging.level', 'INFO'))
    log_format = config.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('logging.file', 'logs/app.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Set up logging
logger = setup_logging()