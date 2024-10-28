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
    """Create standardized mock arguments for testing.
    
    Args:
        overrides (dict, optional): Dictionary of argument overrides. Defaults to None.
        
    Returns:
        MagicMock: Mock arguments object with standard testing values.
    """
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

def create_gui_app(mock_args=None):
    """Create a GUI application instance with mocked arguments.
    
    Args:
        mock_args (MagicMock, optional): Mock arguments to use. If None, creates new ones.
        
    Returns:
        tuple: (AiderVoiceGUI instance, tk.Tk root window)
    """
    if mock_args is None:
        mock_args = create_mock_args()
    with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        app.setup_gui()
        return app, root

def create_buffer_manager(max_size=1024, chunk_size=256, sample_rate=24000):
    """Create an AudioBufferManager instance for testing.
    
    Args:
        max_size (int, optional): Maximum buffer size in bytes. Defaults to 1024.
        chunk_size (int, optional): Size of audio chunks. Defaults to 256.
        sample_rate (int, optional): Audio sample rate. Defaults to 24000.
        
    Returns:
        AudioBufferManager: Configured buffer manager instance.
    """
    return AudioBufferManager(
        max_size=max_size,
        chunk_size=chunk_size,
        sample_rate=sample_rate
    )

def create_performance_monitor(metrics=None):
    """Create a PerformanceMonitor instance for testing.
    
    Args:
        metrics (list, optional): List of metrics to monitor. Defaults to ["cpu", "memory", "latency"].
        
    Returns:
        PerformanceMonitor: Configured monitor instance.
    """
    if metrics is None:
        metrics = ["cpu", "memory", "latency"]
    return PerformanceMonitor(metrics)
