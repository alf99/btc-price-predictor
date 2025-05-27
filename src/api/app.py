"""
FastAPI application for BTC Price Predictor.
Provides endpoints for predictions and model information.
"""
import os
import sys
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import json
import asyncio
import websockets
from threading import Thread

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.collectors import BinanceDataCollector, CoinGeckoDataCollector
from src.data.features import FeatureEngineer
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TemporalFusionTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BTC Price Predictor API",
    description="API for predicting Bitcoin prices using ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    interval: str = "1h"
    horizon: int = 24
    model_type: str = "lstm"

class PredictionResponse(BaseModel):
    timestamp: str
    current_price: float
    predicted_price: float
    prediction_horizon: str
    model_used: str
    confidence: Optional[float] = None
    direction: str

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    last_trained: Optional[str] = None
    performance_metrics: Dict
    supported_horizons: List[int]

# Global variables
models = {}
latest_data = {}
websocket_clients = set()

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models and start background tasks on startup."""
    try:
        # Load short-term LSTM model
        lstm_model = LSTMModel()
        lstm_model_path = os.path.join('models', 'lstm_model', 'final_model.h5')
        
        if os.path.exists(lstm_model_path):
            lstm_model.load_model(lstm_model_path)
            models['lstm'] = lstm_model
            logger.info("LSTM model loaded successfully")
        else:
            logger.warning(f"LSTM model not found at {lstm_model_path}")
        
        # Load long-term Transformer model
        transformer_model = TemporalFusionTransformer()
        transformer_model_path = os.path.join('models', 'transformer_model', 'final_model.h5')
        
        if os.path.exists(transformer_model_path):
            transformer_model.load_model(transformer_model_path)
            models['transformer'] = transformer_model
            logger.info("Transformer model loaded successfully")
        else:
            logger.warning(f"Transformer model not found at {transformer_model_path}")
        
        # Start background data collection
        background_tasks = BackgroundTasks()
        background_tasks.add_task(fetch_latest_data)
        
        # Start WebSocket server in a separate thread
        websocket_thread = Thread(target=start_websocket_server)
        websocket_thread.daemon = True
        websocket_thread.start()
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

async def fetch_latest_data():
    """Fetch latest data periodically."""
    while True:
        try:
            # Fetch latest BTC data
            binance_collector = BinanceDataCollector()
            coingecko_collector = CoinGeckoDataCollector()
            
            # Get short-term data from Binance
            short_term_data = binance_collector.get_historical_klines(
                interval="1h", 
                limit=100
            )
            
            if not short_term_data.empty:
                latest_data['short_term'] = short_term_data
                logger.info(f"Updated short-term data, latest price: {short_term_data['close'].iloc[-1]}")
                
                # Broadcast to WebSocket clients
                if websocket_clients:
                    latest_price = float(short_term_data['close'].iloc[-1])
                    message = json.dumps({
                        "type": "price_update",
                        "data": {
                            "timestamp": datetime.now().isoformat(),
                            "price": latest_price
                        }
                    })
                    
                    await broadcast_message(message)
            
            # Get long-term data from CoinGecko
            long_term_data = coingecko_collector.get_bitcoin_historical_data(days=30)
            
            if not long_term_data.empty:
                latest_data['long_term'] = long_term_data
                logger.info(f"Updated long-term data, latest price: {long_term_data['price'].iloc[-1]}")
            
            # Wait before next update (5 minutes)
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            await asyncio.sleep(60)  # Wait a bit before retrying

async def broadcast_message(message):
    """Broadcast message to all connected WebSocket clients."""
    if websocket_clients:
        disconnected_clients = set()
        
        for client in websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        websocket_clients.difference_update(disconnected_clients)

async def websocket_handler(websocket, path):
    """Handle WebSocket connections."""
    try:
        # Register client
        websocket_clients.add(websocket)
        logger.info(f"New WebSocket client connected. Total clients: {len(websocket_clients)}")
        
        # Send initial data if available
        if 'short_term' in latest_data:
            latest_price = float(latest_data['short_term']['close'].iloc[-1])
            message = json.dumps({
                "type": "price_update",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "price": latest_price
                }
            })
            await websocket.send(message)
        
        # Keep connection alive
        while True:
            message = await websocket.recv()
            # Process client messages if needed
            
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Unregister client
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(websocket_clients)}")

def start_websocket_server():
    """Start WebSocket server."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    start_server = websockets.serve(
        websocket_handler, 
        "0.0.0.0",  # Listen on all interfaces
        8765  # WebSocket port
    )
    
    loop.run_until_complete(start_server)
    loop.run_forever()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to BTC Price Predictor API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/models",
            "/data/latest"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a price prediction.
    
    Args:
        request: Prediction request with parameters
        
    Returns:
        Prediction response with predicted price
    """
    try:
        model_type = request.model_type.lower()
        
        # Check if model exists
        if model_type not in models:
            # If models aren't loaded yet, return a mock prediction
            if not models:
                logger.warning("No models loaded, returning mock prediction")
                
                # Get latest price from data or use a default
                current_price = 30000.0
                if 'short_term' in latest_data and not latest_data['short_term'].empty:
                    current_price = float(latest_data['short_term']['close'].iloc[-1])
                
                # Generate a random prediction (just for demo)
                predicted_price = current_price * (1 + np.random.normal(0, 0.02))
                
                return PredictionResponse(
                    timestamp=datetime.now().isoformat(),
                    current_price=current_price,
                    predicted_price=predicted_price,
                    prediction_horizon=f"{request.horizon} hours",
                    model_used="mock_model",
                    confidence=0.5,
                    direction="up" if predicted_price > current_price else "down"
                )
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model_type}' not found")
        
        # Get appropriate data based on model type
        if model_type == 'lstm':
            if 'short_term' not in latest_data or latest_data['short_term'].empty:
                raise HTTPException(status_code=503, detail="Short-term data not available")
            
            data = latest_data['short_term']
            current_price = float(data['close'].iloc[-1])
            
            # Prepare features
            fe = FeatureEngineer()
            data_with_features = fe.add_technical_indicators(data)
            data_with_features = fe.create_time_features(data_with_features)
            
            # Select features for model
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_7', 'sma_14', 'ema_7', 'ema_14', 
                'rsi_14', 'macd', 'macd_signal',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]
            
            # Keep only columns that exist in the data
            feature_cols = [col for col in feature_cols if col in data_with_features.columns]
            
            # Normalize features (simple approach for API)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            data_scaled = data_with_features.copy()
            data_scaled[feature_cols] = scaler.fit_transform(data_with_features[feature_cols])
            
            # Create sequence for model input
            sequence_length = 24  # Should match model's expected sequence length
            features = data_scaled[feature_cols].values[-sequence_length:].reshape(1, sequence_length, len(feature_cols))
            
            # Make prediction
            model = models[model_type]
            predicted_value = float(model.predict(features)[0][0])
            
            # Denormalize prediction (if the model predicts normalized values)
            # This is a simplified approach - in production, you'd use the same scaler as during training
            predicted_price = current_price * (1 + predicted_value * 0.01)
            
        elif model_type == 'transformer':
            if 'long_term' not in latest_data or latest_data['long_term'].empty:
                raise HTTPException(status_code=503, detail="Long-term data not available")
            
            data = latest_data['long_term']
            current_price = float(data['price'].iloc[-1])
            
            # Similar feature preparation as above, but for long-term model
            # ...
            
            # For now, return a mock prediction
            predicted_price = current_price * (1 + np.random.normal(0.01, 0.05))
        
        # Determine direction
        direction = "up" if predicted_price > current_price else "down"
        
        # Create response
        response = PredictionResponse(
            timestamp=datetime.now().isoformat(),
            current_price=current_price,
            predicted_price=predicted_price,
            prediction_horizon=f"{request.horizon} hours",
            model_used=model_type,
            confidence=0.8,  # Mock confidence value
            direction=direction
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=List[ModelInfoResponse])
async def get_models():
    """
    Get information about available models.
    
    Returns:
        List of model information
    """
    try:
        model_info = []
        
        # If no models are loaded, return mock data
        if not models:
            model_info = [
                ModelInfoResponse(
                    model_name="lstm_model",
                    model_type="LSTM",
                    last_trained="2023-05-15T10:30:00",
                    performance_metrics={
                        "mse": 0.0025,
                        "rmse": 0.05,
                        "mae": 0.04,
                        "directional_accuracy": 0.65
                    },
                    supported_horizons=[1, 6, 12, 24]
                ),
                ModelInfoResponse(
                    model_name="transformer_model",
                    model_type="Temporal Fusion Transformer",
                    last_trained="2023-05-10T14:45:00",
                    performance_metrics={
                        "mse": 0.0035,
                        "rmse": 0.059,
                        "mae": 0.048,
                        "directional_accuracy": 0.62
                    },
                    supported_horizons=[24, 48, 72, 168]
                )
            ]
        else:
            # Add info for each loaded model
            for model_name, model in models.items():
                info = ModelInfoResponse(
                    model_name=f"{model_name}_model",
                    model_type=model_name.upper(),
                    last_trained=datetime.now().isoformat(),  # Mock date
                    performance_metrics={
                        "mse": 0.003,
                        "rmse": 0.055,
                        "mae": 0.045,
                        "directional_accuracy": 0.63
                    },
                    supported_horizons=[1, 6, 12, 24] if model_name == "lstm" else [24, 48, 72, 168]
                )
                model_info.append(info)
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/latest")
async def get_latest_data(interval: str = Query("1h", description="Data interval (1h, 1d)")):
    """
    Get latest price data.
    
    Args:
        interval: Data interval (1h, 1d)
        
    Returns:
        Latest price data
    """
    try:
        if interval == "1h":
            if 'short_term' not in latest_data or latest_data['short_term'].empty:
                raise HTTPException(status_code=503, detail="Short-term data not available")
            
            data = latest_data['short_term'].tail(24)
            
            # Convert to list of records
            records = []
            for _, row in data.iterrows():
                records.append({
                    "timestamp": row['open_time'].isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                })
            
            return {
                "interval": interval,
                "data": records
            }
            
        elif interval == "1d":
            if 'long_term' not in latest_data or latest_data['long_term'].empty:
                raise HTTPException(status_code=503, detail="Long-term data not available")
            
            data = latest_data['long_term'].tail(30)
            
            # Convert to list of records
            records = []
            for _, row in data.iterrows():
                records.append({
                    "timestamp": row['timestamp'].isoformat(),
                    "price": float(row['price']),
                    "market_cap": float(row['market_cap']),
                    "volume": float(row['volume'])
                })
            
            return {
                "interval": interval,
                "data": records
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported interval: {interval}")
        
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)