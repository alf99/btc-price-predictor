"""
Main entry point for BTC Price Predictor application.
"""
import os
import logging
import uvicorn
import argparse
from src.api.app import app as api_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BTC Price Predictor')
    parser.add_argument('--api-port', type=int, default=8000, help='API port')
    parser.add_argument('--api-host', type=str, default='0.0.0.0', help='API host')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Start API server
    logger.info(f"Starting API server on {args.api_host}:{args.api_port}")
    uvicorn.run(
        "src.api.app:app",
        host=args.api_host,
        port=args.api_port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()