"""WebSocket connection management."""

import asyncio
import json
import time
import websockets
from enum import Enum, auto
from .exceptions import WebSocketConnectionError

OPENAI_WEBSOCKET_URL = (
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
)

class ConnectionState(Enum):
    """Enum for WebSocket connection states"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()
    ERROR = auto()
    CLOSING = auto()

class WebSocketManager:
    """Manages WebSocket connection state and monitoring"""
    
    def __init__(self, parent):
        self.parent = parent
        self._state = ConnectionState.DISCONNECTED
        self.last_ping_time = 0
        self.ping_interval = 30
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.monitoring_task = None
        self.log_message = parent.log_message
        self.ws = parent.ws
        self.last_error = None
        self.error_time = 0

    # ... rest of WebSocketManager implementation ...
