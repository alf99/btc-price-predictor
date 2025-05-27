"""
WebSocket utility for BTC Price Predictor.
Handles WebSocket connections for real-time data streaming.
"""
import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional

from src.utils.config import config

logger = logging.getLogger(__name__)

class WebSocketManager:
    """WebSocket manager for handling real-time data streaming."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.host = config.get('api.host', '0.0.0.0')
        self.port = config.get('api.websocket_port', 8765)
        self.clients = set()
        self.server = None
        self.running = False
        self.message_handlers = {}
    
    async def handler(self, websocket, path):
        """
        Handle WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type in self.message_handlers:
                        await self.message_handlers[message_type](websocket, data)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f"Unknown message type: {message_type}"
                        }))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': "Invalid JSON message"
                    }))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        if not self.clients:
            return
        
        message_json = json.dumps(message)
        await asyncio.gather(
            *[client.send(message_json) for client in self.clients],
            return_exceptions=True
        )
    
    async def start_server(self):
        """Start WebSocket server."""
        self.server = await websockets.serve(
            self.handler,
            self.host,
            self.port
        )
        self.running = True
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def stop_server(self):
        """Stop WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.running = False
            logger.info("WebSocket server stopped")
    
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register message handler.
        
        Args:
            message_type: Message type
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")

# Create global WebSocket manager instance
websocket_manager = WebSocketManager()