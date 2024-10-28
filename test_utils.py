"""Test utilities and fixtures for aider_wrapper tests.

This module provides utility functions and classes for testing the aider_wrapper package,
including mock objects, test fixtures, and helper functions for GUI and async testing.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import asyncio
import tkinter as tk
from unittest.mock import MagicMock, patch

from aider_wrapper import (
    AiderVoiceGUI,
    AudioBufferManager,
    PerformanceMonitor,
    WebSocketManager,
    ConnectionState,
)

class AsyncMock(MagicMock):
    """Enhanced mock class that supports async methods and context managers.
    
    This class extends MagicMock to provide proper async method mocking, including
    support for async context managers, iterators, and await operations.
    
    Attributes:
        return_value: The value to return from async calls
        side_effect: Optional side effect function for async calls
    """
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._is_async = True
        self._return_value: Any = kwargs.get('return_value', True)
        self._side_effect: Optional[Union[Exception, callable]] = kwargs.get('side_effect', None)
        self._was_called: bool = False
        
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = super().__call__(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        elif self._side_effect is not None:
            if asyncio.iscoroutine(self._side_effect):
                return await self._side_effect
            return self._side_effect()
        return self._return_value

    async def __aenter__(self) -> 'AsyncMock':
        result = super().__call__()
        if asyncio.iscoroutine(result):
            return await result
        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], 
                       exc_val: Optional[BaseException], 
                       exc_tb: Optional[Any]) -> None:
        pass

    def __await__(self) -> Any:
        async def dummy() -> Any:
            if self._side_effect is not None:
                if asyncio.iscoroutine(self._side_effect):
                    return await self._side_effect
                return self._side_effect()
            return self._return_value
        return dummy().__await__()

    async def __aiter__(self) -> 'AsyncMock':
        return self

    async def __anext__(self) -> None:
        raise StopAsyncIteration

def create_mock_args(overrides: Optional[Dict[str, Any]] = None, 
                    validate: bool = True) -> MagicMock:
    """Create standardized mock arguments for testing.
    
    Args:
        overrides: Optional dictionary of argument overrides
        
    Returns:
        MagicMock: Mock arguments object with standard testing values
        
    Example:
        >>> args = create_mock_args({'model': 'gpt-3.5-turbo'})
        >>> assert args.model == 'gpt-3.5-turbo'
    """
    args = MagicMock()
    
    # Standard argument values
    default_args = {
        'voice_only': False,
        'instructions': None,
        'clipboard': False,
        'chat_mode': "code",
        'suggest_shell_commands': False,
        'model': "gpt-4",
        'gui': True,
        'auto': False,
        'api_key': "test_key",
        'verbose': False,
        'temperature': 0.7,
        'max_tokens': 2000,
        'files': [],
        'timeout': 30,
        'retry_count': 3,
        'buffer_size': 1024,
    }
    
    # Set default values
    for key, value in default_args.items():
        setattr(args, key, value)
    
    # Apply any overrides
    if overrides:
        for key, value in overrides.items():
            if validate and key not in default_args:
                raise ValueError(f"Unknown argument override: {key}")
            setattr(args, key, value)
            
        # Validate argument combinations
        if validate:
            if getattr(args, 'voice_only', False) and getattr(args, 'gui', False):
                raise ValueError("Cannot use voice_only with gui=True")
            if getattr(args, 'temperature', 0) < 0 or getattr(args, 'temperature', 0) > 1:
                raise ValueError("Temperature must be between 0 and 1")
            if getattr(args, 'max_tokens', 0) < 1:
                raise ValueError("max_tokens must be positive")
            
    return args

def create_gui_app(mock_args: Optional[MagicMock] = None,
                  setup_websocket: bool = False,
                  validate_state: bool = True) -> Tuple[AiderVoiceGUI, tk.Tk]:
    """Create a GUI application instance with mocked arguments and optional WebSocket setup.
    
    Args:
        mock_args: Optional mock arguments to use. If None, creates new ones.
        setup_websocket: Whether to setup and mock WebSocket connection
        
    Returns:
        Tuple containing:
            - AiderVoiceGUI instance
            - tk.Tk root window
            
    Example:
        >>> app, root = create_gui_app(setup_websocket=True)
        >>> assert app.ws_manager.connection_state == ConnectionState.CONNECTED
    """
    if mock_args is None:
        mock_args = create_mock_args()
        
    with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        app.setup_gui()
        
        if setup_websocket:
            app.ws = AsyncMock()
            app.ws_manager = WebSocketManager(app)
            app.ws_manager.connection_state = ConnectionState.CONNECTED
            
        if validate_state:
            # Validate GUI state
            if not hasattr(app, 'main_frame') or not app.main_frame:
                raise RuntimeError("GUI initialization failed - main_frame missing")
            if not hasattr(app, 'input_text') or not app.input_text:
                raise RuntimeError("GUI initialization failed - input_text missing")
            if not hasattr(app, 'output_text') or not app.output_text:
                raise RuntimeError("GUI initialization failed - output_text missing")
                
        return app, root

def create_buffer_manager(max_size: int = 1024,
                        chunk_size: int = 256,
                        sample_rate: int = 24000,
                        validate: bool = True) -> AudioBufferManager:
    """Create an AudioBufferManager instance for testing.
    
    Args:
        max_size: Maximum buffer size in bytes
        chunk_size: Size of audio chunks
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Configured AudioBufferManager instance
        
    Raises:
        ValueError: If any parameters are invalid
    """
    if validate:
        if max_size <= 0 or chunk_size <= 0 or sample_rate <= 0:
            raise ValueError("Buffer parameters must be positive")
        if chunk_size > max_size:
            raise ValueError("Chunk size cannot exceed max size")
        if sample_rate not in {8000, 16000, 24000, 44100, 48000}:
            raise ValueError("Invalid sample rate")
        if chunk_size % 2 != 0:
            raise ValueError("Chunk size must be even")
        
    return AudioBufferManager(
        max_size=max_size,
        chunk_size=chunk_size,
        sample_rate=sample_rate
    )

def create_performance_monitor(metrics: Optional[List[str]] = None,
                            log_interval: int = 5) -> PerformanceMonitor:
    """Create a PerformanceMonitor instance for testing.
    
    Args:
        metrics: List of metrics to monitor
        log_interval: Interval between metric logs in seconds
        
    Returns:
        Configured PerformanceMonitor instance
        
    Raises:
        ValueError: If metrics list is empty or contains invalid metrics
    """
    if metrics is None:
        metrics = ["cpu", "memory", "latency"]
    elif not metrics:
        raise ValueError("Metrics list cannot be empty")
        
    valid_metrics = {"cpu", "memory", "latency", "network", "disk"}
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        raise ValueError(f"Invalid metrics: {invalid_metrics}")
        
    return PerformanceMonitor(metrics, log_interval=log_interval)

def create_websocket_manager(parent: Any,
                           initial_state: ConnectionState = ConnectionState.DISCONNECTED
                           ) -> WebSocketManager:
    """Create a WebSocketManager instance for testing.
    
    Args:
        parent: Parent object (usually AiderVoiceGUI instance)
        initial_state: Initial connection state
        
    Returns:
        Configured WebSocketManager instance
    """
    manager = WebSocketManager(parent)
    manager.connection_state = initial_state
    return manager
