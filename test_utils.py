"""Test utilities and fixtures for aider_wrapper tests."""

import tkinter as tk
from unittest.mock import MagicMock, patch

from aider_wrapper import (
    AiderVoiceGUI,
    AudioBufferManager,
    PerformanceMonitor,
)

class AsyncMock(MagicMock):
    """Mock class that supports async methods and special method handling."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.side_effect = lambda: True
        
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __await__(self):
        async def dummy():
            return self
        return dummy().__await__()

    async def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

def create_mock_args(overrides=None):
    """Create standardized mock arguments for testing"""
    args = MagicMock()
    args.voice_only = False
    args.instructions = None
    args.clipboard = False
    args.chat_mode = "code"
    args.suggest_shell_commands = False
    args.model = "gpt-4"
    args.gui = True
    args.auto = False
    args.api_key = "test_key"
    args.verbose = False
    args.temperature = 0.7
    args.max_tokens = 2000
    args.files = []
    
    if overrides:
        for key, value in overrides.items():
            setattr(args, key, value)
            
    return args
    """Create standardized mock arguments for testing."""
    args = MagicMock()
    args.voice_only = False
    args.instructions = None
    args.clipboard = False
    args.chat_mode = "code"
    args.suggest_shell_commands = False
    args.model = "gpt-4"
    args.gui = True
    args.auto = False
    args.api_key = "test_key"
    args.verbose = False
    args.temperature = 0.7
    args.max_tokens = 2000
    args.files = []
    return args

def create_gui_app(mock_args=None):
    """Create a GUI application instance with mocked arguments"""
    if mock_args is None:
        mock_args = create_mock_args()
    with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        app.setup_gui()
        return app, root
    """Create a GUI application instance with mocked arguments."""
    mock_args = create_mock_args()
    with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        app.setup_gui()
        return app, root

def create_buffer_manager():
    """Create an AudioBufferManager instance for testing."""
    return AudioBufferManager(
        max_size=1024,
        chunk_size=256,
        sample_rate=24000
    )

def create_performance_monitor():
    """Create a PerformanceMonitor instance for testing."""
    metrics = ["cpu", "memory", "latency"]
    return PerformanceMonitor(metrics)
def create_buffer_manager(max_size=1024, chunk_size=256, sample_rate=24000):
    """Create an AudioBufferManager instance for testing"""
    return AudioBufferManager(
        max_size=max_size,
        chunk_size=chunk_size,
        sample_rate=sample_rate
    )

def create_performance_monitor(metrics=None):
    """Create a PerformanceMonitor instance for testing"""
    if metrics is None:
        metrics = ["cpu", "memory", "latency"]
    return PerformanceMonitor(metrics)
