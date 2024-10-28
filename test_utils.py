import tkinter as tk
from unittest.mock import MagicMock, AsyncMock
from aider_wrapper import AiderVoiceGUI

def create_mock_args():
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
    return args

def create_gui_app():
    """Create a GUI application instance with mocked arguments"""
    mock_args = create_mock_args()
    with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        app.setup_gui()
        return app, root

def create_buffer_manager():
    """Create an AudioBufferManager instance for testing"""
    return AudioBufferManager(
        max_size=1024,
        chunk_size=256,
        sample_rate=24000
    )

def create_performance_monitor():
    """Create a PerformanceMonitor instance for testing"""
    metrics = ["cpu", "memory", "latency"]
    return PerformanceMonitor(metrics)

class AsyncMock(MagicMock):
    """Mock class that supports async methods and special method handling"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.side_effect = lambda: True
        
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

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
