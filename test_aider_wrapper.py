import unittest
import asyncio
import tkinter as tk
from unittest.mock import MagicMock, patch
import sys
from aider_wrapper import AiderVoiceGUI, AudioBufferManager, PerformanceMonitor, WebSocketManager

class AsyncMock(MagicMock):
    """Mock class that supports async methods and special method handling"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default return values for special methods
        self.__bool__.return_value = True
        
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

class TestAiderVoiceGUI(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        # Mock argument parser
        patcher = patch('argparse.ArgumentParser.parse_args')
        self.mock_parse_args = patcher.start()
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
        self.assertIn("🎤 " + test_text, transcription)

    @patch('websockets.connect', new_callable=AsyncMock)
    def test_connect_websocket(self, mock_connect):
        """Test websocket connection and message handling"""
        # Create async mock for websocket with proper boolean behavior
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.__bool__.return_value = True
        mock_connect.return_value = mock_ws

        async def run_test():
            # Start message handling tasks
            message_task = asyncio.create_task(self.app.handle_websocket_messages())
            queue_task = asyncio.create_task(self.app.process_audio_queue())

            try:
                result = await self.app.connect_websocket()
                self.assertTrue(result)
                self.assertEqual(self.app.ws, mock_ws)
                
                # Verify session.update was sent with correct data
                mock_ws.send.assert_called_once()
                call_args = mock_ws.send.call_args[0][0]
                self.assertIn("session.update", call_args)
                self.assertIn("model", call_args)
                
                # Additional assertions to verify connection state
                self.assertTrue(self.app.ws is not None)
                self.assertFalse(self.app.response_active)
                self.assertIsNone(self.app.last_transcript_id)
                self.assertEqual(len(self.app.audio_buffer), 0)
                
            finally:
                # Clean up tasks
                message_task.cancel()
                queue_task.cancel()
                try:
                    await message_task
                    await queue_task
                except asyncio.CancelledError:
                    pass

        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    @patch('websockets.connect', new_callable=AsyncMock)
    def test_connect_websocket_failure(self, mock_connect):
        """Test websocket connection failure handling"""
        mock_connect.side_effect = Exception("Connection failed")

        async def run_test():
            result = await self.app.connect_websocket()
            self.assertFalse(result)
            self.assertIsNone(self.app.ws)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    @patch('websockets.connect', new_callable=AsyncMock)
    async def test_websocket_timeout(self, mock_connect):
        """Test websocket timeout handling"""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_connect.return_value = mock_ws

        result = await self.app.connect_websocket()
        self.assertFalse(result)

    @patch('websockets.connect', new_callable=AsyncMock)
    async def test_websocket_close(self, mock_connect):
        """Test websocket close handling"""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()
        mock_connect.return_value = mock_ws

        result = await self.app.connect_websocket()
        self.assertTrue(result)
        await self.app.ws.close()
        mock_ws.close.assert_called_once()
        """Test websocket connection and message handling"""
        # Create async mock for websocket with proper boolean behavior
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.__bool__.return_value = True
        mock_connect.return_value = mock_ws

        async def run_test():
            # Start message handling tasks
            message_task = asyncio.create_task(self.app.handle_websocket_messages())
            queue_task = asyncio.create_task(self.app.process_audio_queue())

            try:
                result = await self.app.connect_websocket()
                self.assertTrue(result)
                self.assertEqual(self.app.ws, mock_ws)
                
                # Verify session.update was sent with correct data
                mock_ws.send.assert_called_once()
                call_args = mock_ws.send.call_args[0][0]
                self.assertIn("session.update", call_args)
                self.assertIn("model", call_args)
                
                # Additional assertions to verify connection state
                self.assertTrue(self.app.ws is not None)
                self.assertFalse(self.app.response_active)
                self.assertIsNone(self.app.last_transcript_id)
                self.assertEqual(len(self.app.audio_buffer), 0)
                
            finally:
                # Clean up tasks
                message_task.cancel()
                queue_task.cancel()
                try:
                    await message_task
                    await queue_task
                except asyncio.CancelledError:
                    pass

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

class TestVoiceCommandProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.parent = MagicMock()
        self.processor = VoiceCommandProcessor(self.parent)

    def test_init(self):
        """Test initialization"""
        self.assertIsInstance(self.processor.commands, list)
        self.assertEqual(len(self.processor.commands), 0)

    def test_preprocess_command(self):
        """Test command preprocessing"""
        # Test stripping whitespace
        self.assertEqual(self.processor.preprocess_command("  test  "), "test")
        # Test converting to lowercase
        self.assertEqual(self.processor.preprocess_command("TEST"), "test")
        # Test combined effects
        self.assertEqual(self.processor.preprocess_command("  TEST  "), "test")

    def test_validate_command(self):
        """Test command validation"""
        # Test empty command
        self.assertFalse(self.processor.validate_command(""))
        self.assertFalse(self.processor.validate_command("   "))
        # Test valid command
        self.assertTrue(self.processor.validate_command("test command"))


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

if __name__ == '__main__':
    unittest.main(verbosity=2)
