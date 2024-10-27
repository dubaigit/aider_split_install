import unittest
import asyncio
import tkinter as tk
from unittest.mock import MagicMock, patch
import sys
from aider_wrapper import AiderVoiceGUI, AudioBufferManager, PerformanceMonitor, WebSocketManager

class AsyncMock(MagicMock):
    """Mock class that supports async methods"""
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

class TestAiderVoiceGUI(unittest.TestCase):
        self.addCleanup(patcher.stop)

        mock_args = MagicMock()
        mock_args.voice_only = False
        mock_args.instructions = None
        mock_args.clipboard = False
        mock_args.chat_mode = "code"
        mock_args.suggest_shell_commands = False
        mock_args.model = None
        mock_args.gui = True
        mock_args.auto = False
        self.mock_parse_args.return_value = mock_args

        self.root = tk.Tk()
        self.app = AiderVoiceGUI(self.root)
        # Force GUI setup
        self.app.setup_gui()

    def tearDown(self):
        """Clean up after each test"""
        self.root.destroy()

    def test_init(self):
        """Test initialization of GUI components"""
        self.assertIsNotNone(self.app.root)
        self.assertIsNotNone(self.app.main_frame)
        self.assertIsNotNone(self.app.status_label)
        self.assertIsNotNone(self.app.input_text)
        self.assertIsNotNone(self.app.output_text)
        self.assertIsNotNone(self.app.transcription_text)
        self.assertIsNotNone(self.app.issues_text)

    def test_log_message(self):
        """Test logging messages to output text"""
        test_message = "Test message"
        self.app.log_message(test_message)
        output_text = self.app.output_text.get("1.0", tk.END).strip()
        self.assertEqual(output_text, test_message)

    def test_update_transcription(self):
        """Test updating transcription text"""
        test_text = "Test transcription"
        self.app.update_transcription(test_text, is_assistant=False)
        transcription = self.app.transcription_text.get("1.0", tk.END).strip()
        self.assertIn(test_text, transcription)

    @patch('websockets.connect')
    def test_connect_websocket(self, mock_connect):
        """Test websocket connection"""
        # Create async mock for websocket
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_connect.return_value = mock_ws

        async def run_test():
            result = await self.app.connect_websocket()
            self.assertTrue(result)
            self.assertEqual(self.app.ws, mock_ws)
            mock_ws.send.assert_called_once()  # Verify session.update was sent

        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

class TestAudioBufferManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.buffer_manager = AudioBufferManager(
            max_size=1024,
            chunk_size=256,
            sample_rate=24000
        )

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.buffer_manager.max_size, 1024)
        self.assertEqual(self.buffer_manager.chunk_size, 256)
        self.assertEqual(self.buffer_manager.sample_rate, 24000)
        self.assertEqual(len(self.buffer_manager.buffer), 0)

    def test_get_usage(self):
        """Test buffer usage calculation"""
        self.buffer_manager.buffer = bytearray(512)
        self.assertEqual(self.buffer_manager.get_usage(), 0.5)

class TestPerformanceMonitor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.metrics = ["cpu", "memory", "latency"]
        self.monitor = PerformanceMonitor(self.metrics)

    def test_init(self):
        """Test initialization"""
        self.assertEqual(list(self.monitor.metrics.keys()), self.metrics)
        for metric in self.metrics:
            self.assertEqual(self.monitor.metrics[metric], [])

    def test_update(self):
        """Test metric updates"""
        self.monitor.update("cpu", 50)
        self.assertEqual(self.monitor.metrics["cpu"], [50])

    def test_get_metrics(self):
        """Test getting metric averages"""
        self.monitor.update("cpu", 50)
        self.monitor.update("cpu", 60)
        metrics = self.monitor.get_metrics()
        self.assertEqual(metrics["cpu"], 55)

class TestWebSocketManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.parent = MagicMock()
        self.manager = WebSocketManager(self.parent)

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.manager.connection_state, "disconnected")
        self.assertEqual(self.manager.reconnect_attempts, 0)
        self.assertEqual(self.manager.max_reconnect_attempts, 5)

    def test_attempt_reconnect(self):
        """Test reconnection attempts"""
        # Mock the connect_websocket coroutine
        async def mock_connect():
            return True
        self.parent.connect_websocket = mock_connect
        
        async def run_test():
            await self.manager.attempt_reconnect()
            self.assertEqual(self.manager.connection_state, "connected")
            self.assertEqual(self.manager.reconnect_attempts, 0)

        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

def run_tests():
    """Run the tests with proper setup"""
    # Store original argv
    orig_argv = sys.argv
    
    try:
        # Modify argv to remove -v if present
        sys.argv = [arg for arg in sys.argv if arg != '-v']
        unittest.main(verbosity=2, exit=False)
    finally:
        # Restore original argv
        sys.argv = orig_argv

if __name__ == '__main__':
    run_tests()
