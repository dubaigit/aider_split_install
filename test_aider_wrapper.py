import unittest
import asyncio
import tkinter as tk
from unittest.mock import MagicMock, patch
import sys
import json
import time
from queue import Queue
import websockets
import pyaudio
from aider_wrapper import (
    AiderVoiceGUI,
    AudioBufferManager,
    AudioProcessingError,
    PerformanceMonitor,
    WebSocketManager,
    VoiceCommandProcessor,
    ClipboardManager,
    ConnectionState,
)

import unittest
import asyncio
import tkinter as tk
from unittest.mock import MagicMock, patch
import sys
import json
import time
from queue import Queue
import websockets
import pyaudio
from aider_wrapper import (
    AiderVoiceGUI,
    AudioBufferManager,
    AudioProcessingError,
    PerformanceMonitor,
    WebSocketManager,
    VoiceCommandProcessor,
    ClipboardManager,
)

class AsyncMock(MagicMock):
    """Mock class that supports async methods and special method handling"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set default return values for special methods
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

# Test fixtures and utilities
@classmethod
def setUpClass(cls):
    """Set up shared test fixtures"""
    cls.mock_args = cls.create_mock_args()
    
@staticmethod
def mock_args():
    """Fixture for mocked command line arguments"""
    args = MagicMock()
    args.voice_only = False
    args.instructions = None
    args.clipboard = False
    args.chat_mode = "code"
    args.suggest_shell_commands = False
    args.model = None
    args.gui = True
    args.auto = False
    return args

@classmethod
def create_gui_app(cls, mock_args):
    """Fixture for GUI application instance"""
    with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        app.setup_gui()
        yield app
        root.destroy()


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
        mock_ws.side_effect = lambda: True
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

        async def run_test():
            result = await self.app.connect_websocket()
            self.assertFalse(result)

        await run_async_test(run_test())

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
        mock_ws.side_effect = lambda: True
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

@classmethod
def create_buffer_manager(cls):
    """Fixture for AudioBufferManager instance"""
    return AudioBufferManager(
        max_size=1024,
        chunk_size=256,
        sample_rate=24000
    )

class TestAudioBufferManager(unittest.TestCase):
    """Test suite for audio buffer management functionality"""
    
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
        self.assertEqual(self.buffer_manager.stats["drops"], 0)
        self.assertEqual(self.buffer_manager.stats["overflows"], 0)

    def test_get_usage(self):
        """Test buffer usage calculation"""
        self.buffer_manager.buffer = bytearray(512)
        self.assertEqual(self.buffer_manager.get_usage(), 0.5)
        
        self.buffer_manager.buffer = bytearray(1024)
        self.assertEqual(self.buffer_manager.get_usage(), 1.0)
        
        self.buffer_manager.buffer = bytearray()
        self.assertEqual(self.buffer_manager.get_usage(), 0.0)

    def test_get_chunks_empty_queue(self):
        """Test getting chunks from empty queue"""
        test_queue = Queue()
        chunks = self.buffer_manager.get_chunks(test_queue)
        self.assertEqual(len(chunks), 0)

    def test_get_chunks_with_data(self):
        """Test getting chunks with valid data"""
        test_queue = Queue()
        test_data = [b"test1", b"test2", b"test3"]
        for data in test_data:
            test_queue.put(data)
            
        chunks = self.buffer_manager.get_chunks(test_queue)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks, test_data)

    def test_get_chunks_overflow(self):
        """Test chunk overflow handling"""
        test_queue = Queue()
        # Put data larger than max_size
        test_queue.put(b"x" * (self.buffer_manager.max_size + 100))
        
        chunks = self.buffer_manager.get_chunks(test_queue)
        self.assertEqual(len(chunks), 0)
        self.assertEqual(self.buffer_manager.stats["overflows"], 1)

    def test_combine_chunks(self):
        """Test chunk combination"""
        test_chunks = [b"test1", b"test2", b"test3"]
        combined = self.buffer_manager.combine_chunks(test_chunks)
        self.assertEqual(combined, b"test1test2test3")

    def test_combine_chunks_error(self):
        """Test error handling in combine_chunks"""
        test_chunks = [b"test1", None, b"test3"]
        with self.assertRaises(AudioProcessingError):
            self.buffer_manager.combine_chunks(test_chunks)
        self.assertEqual(self.buffer_manager.stats["drops"], 1)

@classmethod
def create_performance_monitor(cls):
    """Fixture for PerformanceMonitor instance"""
    metrics = ["cpu", "memory", "latency"]
    return PerformanceMonitor(metrics)

class TestPerformanceMonitor(unittest.TestCase):
    """Test suite for performance monitoring functionality"""
    
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
        self.mock_ws = AsyncMock()
        self.mock_ws.ping = AsyncMock()
        self.parent.ws = self.mock_ws

    async def asyncSetUp(self):
        """Set up async test environment"""
        self.manager.connection_state = ConnectionState.DISCONNECTED
        self.manager.reconnect_attempts = 0
        self.manager.log_message.reset_mock()

    async def asyncTearDown(self):
        """Clean up async test environment"""
        self.manager.connection_state = ConnectionState.DISCONNECTED
        await asyncio.sleep(0.1)  # Allow pending tasks to complete

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.manager.connection_state, ConnectionState.DISCONNECTED)
        self.assertEqual(self.manager.reconnect_attempts, 0)
        self.assertEqual(self.manager.max_reconnect_attempts, 5)
        self.assertEqual(self.manager.ping_interval, 30)

    async def test_valid_state_transitions(self):
        """Test valid WebSocket state transitions"""
        await self.asyncSetUp()
        
        # Test DISCONNECTED -> CONNECTING
        self.manager.connection_state = ConnectionState.CONNECTING
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTING)
        self.manager.log_message.assert_called_with(
            "🔄 WebSocket state transition: DISCONNECTED -> CONNECTING\n"
            "Reason: Initial connection attempt\n"
        )
        
        # Test CONNECTING -> CONNECTED
        self.manager.log_message.reset_mock()
        self.manager.connection_state = ConnectionState.CONNECTED
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        self.manager.log_message.assert_called_with(
            "✅ WebSocket state transition: CONNECTING -> CONNECTED\n"
            "Reason: Connection established successfully\n"
        )
        
        await self.asyncTearDown()

    async def test_invalid_state_transitions(self):
        """Test invalid WebSocket state transitions"""
        await self.asyncSetUp()
        
        # Test invalid transition: DISCONNECTED -> CONNECTED
        with self.assertRaises(ValueError) as cm:
            self.manager.connection_state = ConnectionState.CONNECTED
        self.assertIn("Invalid state transition", str(cm.exception))
        
        # Test invalid transition: CONNECTED -> CONNECTING
        self.manager.connection_state = ConnectionState.CONNECTING
        self.manager.connection_state = ConnectionState.CONNECTED
        with self.assertRaises(ValueError) as cm:
            self.manager.connection_state = ConnectionState.CONNECTING
        self.assertIn("Invalid state transition", str(cm.exception))
        
        await self.asyncTearDown()

    async def test_reconnection_state_tracking(self):
        """Test reconnection attempt state tracking"""
        await self.asyncSetUp()
        
        # Test reconnection attempt counting
        self.manager.connection_state = ConnectionState.CONNECTING
        self.manager.connection_state = ConnectionState.RECONNECTING
        self.assertEqual(self.manager.reconnect_attempts, 1)
        
        # Test reconnection attempt reset on successful connection
        self.manager.connection_state = ConnectionState.CONNECTED
        self.assertEqual(self.manager.reconnect_attempts, 0)
        
        await self.asyncTearDown()

    async def test_check_connection_success(self):
        """Test successful connection check"""
        self.manager.connection_state = ConnectionState.CONNECTED
        await self.manager.check_connection()
        self.mock_ws.ping.assert_called_once()
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)

    async def test_check_connection_failure(self):
        """Test connection check failure"""
        self.manager.connection_state = ConnectionState.CONNECTED
        self.mock_ws.ping.side_effect = websockets.exceptions.WebSocketException()
        await self.manager.check_connection()
        self.assertEqual(self.manager.connection_state, ConnectionState.DISCONNECTED)

    async def test_monitor_connection(self):
        """Test connection monitoring"""
        self.manager.check_connection = AsyncMock()
        self.manager.attempt_reconnect = AsyncMock()
        
        # Test monitoring connected state
        self.manager.connection_state = ConnectionState.CONNECTED
        self.manager.last_ping_time = 0
        monitor_task = asyncio.create_task(self.manager.monitor_connection())
        
        try:
            # Test connected state monitoring
            await asyncio.sleep(0.1)
            self.manager.check_connection.assert_called_once()
            
            # Test state transitions
            self.manager.connection_state = ConnectionState.DISCONNECTED
            await asyncio.sleep(0.1)
            self.manager.attempt_reconnect.assert_called_once()
            
            # Test reconnection attempts limit
            self.manager.connection_state = ConnectionState.FAILED
            self.manager.reconnect_attempts = self.manager.max_reconnect_attempts
            await asyncio.sleep(0.1)
            self.assertEqual(self.manager.attempt_reconnect.call_count, 1)
            
            # Verify final state
            self.assertEqual(self.manager.connection_state, ConnectionState.FAILED)
            
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def test_attempt_reconnect_success(self):
        """Test successful reconnection attempt"""
        async def mock_connect():
            return True
        self.parent.connect_websocket = mock_connect
        
        await self.manager.attempt_reconnect()
        self.assertEqual(self.manager.connection_state, ConnectionState.CONNECTED)
        self.assertEqual(self.manager.reconnect_attempts, 0)

    async def test_attempt_reconnect_failure(self):
        """Test failed reconnection attempt"""
        async def mock_connect():
            raise websockets.exceptions.WebSocketException()
        self.parent.connect_websocket = mock_connect
        
        await self.manager.attempt_reconnect()
        self.assertEqual(self.manager.connection_state, ConnectionState.DISCONNECTED)
        self.assertEqual(self.manager.reconnect_attempts, 1)

    async def test_max_reconnect_attempts(self):
        """Test maximum reconnection attempts"""
        self.manager.reconnect_attempts = self.manager.max_reconnect_attempts
        await self.manager.attempt_reconnect()
        self.parent.log_message.assert_called_with("❌ Max reconnection attempts reached")

    def run_async_test(self, coro):
        """Helper method to run async tests"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_check_connection_sync(self):
        """Test check_connection using sync wrapper"""
        self.run_async_test(self.test_check_connection_success())
        self.run_async_test(self.test_check_connection_failure())

    def test_monitor_connection_sync(self):
        """Test monitor_connection using sync wrapper"""
        async def test_coro():
            # Setup
            self.manager.check_connection = AsyncMock()
            self.manager.attempt_reconnect = AsyncMock()
            
            try:
                # Test monitoring connected state
                self.manager.connection_state = ConnectionState.CONNECTED
                self.manager.last_ping_time = 0
                monitor_task = asyncio.create_task(self.manager.monitor_connection())
                
                # Allow monitor to run briefly
                await asyncio.sleep(0.1)
                
                # Verify behavior
                self.manager.check_connection.assert_called_once()
                
                # Test monitoring disconnected state
                self.manager.connection_state = ConnectionState.DISCONNECTED
                await asyncio.sleep(0.1)
                self.manager.attempt_reconnect.assert_called_once()
                
                # Test failed state behavior
                self.manager.connection_state = ConnectionState.FAILED
                self.manager.reconnect_attempts = self.manager.max_reconnect_attempts
                await asyncio.sleep(0.1)
                self.assertEqual(self.manager.attempt_reconnect.call_count, 1)
                
            finally:
                # Cleanup
                if 'monitor_task' in locals():
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass

        # Run the async test in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_coro())
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_attempt_reconnect_sync(self):
        """Test attempt_reconnect using sync wrapper"""
        async def test_success():
            await self.test_attempt_reconnect_success()
            
        async def test_failure():
            await self.test_attempt_reconnect_failure()
            
        self.run_async_test(test_success())
        self.run_async_test(test_failure())

class TestAudioProcessing(unittest.TestCase):
    """Test audio processing functionality"""

    def setUp(self):
        """Set up test environment"""
        self.root = tk.Tk()
        self.app = AiderVoiceGUI(self.root)
        self.chunk_size = 1024
        self.sample_rate = 24000
        self.test_audio = b'\x00\x00' * self.chunk_size  # Silent audio chunk

    def tearDown(self):
        """Clean up after tests"""
        self.root.destroy()

    def test_handle_mic_input(self):
        """Test microphone input handling"""
        async def run_test():
            # Test normal input
            result = await self.app.handle_mic_input(
                self.test_audio,
                self.chunk_size,
                {},
                0
            )
            self.assertEqual(result, (None, pyaudio.paContinue))

            # Test when mic is suppressed
            self.app.mic_on_at = time.time() + 1000  # Future time
            result = await self.app.handle_mic_input(
                self.test_audio,
                self.chunk_size,
                {},
                0
            )
            self.assertEqual(result, (None, pyaudio.paContinue))
            self.assertFalse(self.app.mic_active)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

    def test_handle_speaker_output(self):
        """Test speaker output handling"""
        async def run_test():
            # Test with empty buffer
            result = await self.app.handle_speaker_output(
                None,
                self.chunk_size,
                {},
                0
            )
            self.assertEqual(len(result[0]), self.chunk_size * 2)  # 16-bit audio
            self.assertEqual(result[1], pyaudio.paContinue)

            # Test with data in buffer
            test_data = b'\x01\x00' * self.chunk_size
            self.app.audio_buffer = bytearray(test_data)
            result = await self.app.handle_speaker_output(
                None,
                self.chunk_size,
                {},
                0
            )
            self.assertEqual(result[0], test_data)
            self.assertEqual(result[1], pyaudio.paContinue)
            self.assertEqual(len(self.app.audio_buffer), 0)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

    @patch('websockets.connect', new_callable=AsyncMock)
    async def test_send_audio_chunk(self, mock_connect):
        """Test sending audio chunks to websocket"""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        self.app.ws = mock_ws

        # Test sending valid chunk
        await self.app.send_audio_chunk(self.test_audio)
        mock_ws.send.assert_called_once()
        
        # Verify correct JSON format
        call_args = mock_ws.send.call_args[0][0]
        data = json.loads(call_args)
        self.assertEqual(data['type'], 'input_audio_buffer.append')
        self.assertTrue('audio' in data)

        # Test sending empty chunk
        mock_ws.send.reset_mock()
        await self.app.send_audio_chunk(b'')
        mock_ws.send.assert_not_called()

    def test_audio_buffer_overflow(self):
        """Test audio buffer overflow handling"""
        async def run_test():
            # Create buffer manager with small max size for testing
            buffer_manager = AudioBufferManager(
                max_size=1024,  # 1KB max buffer
                chunk_size=256,
                sample_rate=24000
            )
            
            # Create test data larger than buffer
            test_data = b"\x00" * 2048  # 2KB of data
            
            # Split into chunks
            chunks = [test_data[i:i+256] for i in range(0, len(test_data), 256)]
            
            # Create queue and add chunks
            test_queue = Queue()
            for chunk in chunks:
                test_queue.put(chunk)
            
            # Get chunks with overflow protection
            received_chunks = buffer_manager.get_chunks(test_queue)
            
            # Verify overflow was handled
            self.assertTrue(len(received_chunks) < len(chunks))
            self.assertEqual(buffer_manager.stats["overflows"], 1)
            
            # Verify buffer usage
            combined_data = buffer_manager.combine_chunks(received_chunks)
            self.assertLessEqual(len(combined_data), buffer_manager.max_size)

            # Run the test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_test())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_test())
        finally:
            loop.close()

class TestGUIEventHandlers(unittest.TestCase):
    """Test GUI event handlers and interactions"""
    
    def setUp(self):
        """Set up test environment"""
        self.root = tk.Tk()
        self.app = AiderVoiceGUI(self.root)
        self.app.setup_gui()

    def tearDown(self):
        """Clean up after tests"""
        self.root.destroy()

    def test_browse_files(self):
        """Test browse files button handler"""
        with patch('tkinter.filedialog.askopenfilenames') as mock_dialog:
            # Mock file selection
            test_files = ['/path/test1.py', '/path/test2.py']
            mock_dialog.return_value = test_files
            
            # Trigger browse files
            self.app.browse_files()
            
            # Verify files were added
            for file in test_files:
                self.assertIn(file, self.app.interface_state['files'])
                
            # Verify listbox was updated
            listbox_files = self.app.files_listbox.get(0, tk.END)
            for file in test_files:
                self.assertIn(file, listbox_files)

    def test_remove_selected_file(self):
        """Test remove file button handler"""
        # Add test file
        test_file = '/path/test.py'
        self.app.interface_state['files'][test_file] = None
        self.app.files_listbox.insert(tk.END, test_file)
        
        # Select and remove file
        self.app.files_listbox.selection_set(0)
        self.app.remove_selected_file()
        
        # Verify file was removed
        self.assertNotIn(test_file, self.app.interface_state['files'])
        self.assertEqual(self.app.files_listbox.size(), 0)

    def test_use_clipboard_content(self):
        """Test clipboard button handler"""
        with patch('pyperclip.paste') as mock_paste:
            test_content = "Test clipboard content"
            mock_paste.return_value = test_content
            
            self.app.use_clipboard_content()
            
            # Verify content was inserted
            input_content = self.app.input_text.get('1.0', tk.END).strip()
            self.assertEqual(input_content, test_content)

    def test_send_input_text(self):
        """Test send button handler"""
        test_input = "Test input content"
        self.app.input_text.insert('1.0', test_input)
        
        # Mock log_message to verify it was called
        self.app.log_message = MagicMock()
        
        self.app.send_input_text()
        
        self.app.log_message.assert_called_with("Processing input...")

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcuts"""
        # Mock methods that would be triggered by shortcuts
        self.app.check_all_issues = MagicMock()
        self.app.browse_files = MagicMock()
        self.app.use_clipboard_content = MagicMock()
        self.app.send_input_text = MagicMock()
        self.app.stop_voice_control = MagicMock()
        
        # Test Control-r
        self.root.event_generate('<Control-r>')
        self.app.check_all_issues.assert_called_once()
        
        # Test Control-a
        self.root.event_generate('<Control-a>')
        self.app.browse_files.assert_called_once()
        
        # Test Control-v
        self.root.event_generate('<Control-v>')
        self.app.use_clipboard_content.assert_called_once()
        
        # Test Control-s
        self.root.event_generate('<Control-s>')
        self.app.send_input_text.assert_called_once()
        
        # Test Escape
        self.root.event_generate('<Escape>')
        self.app.stop_voice_control.assert_called_once()

    def test_update_transcription(self):
        """Test transcription updates"""
        # Test user transcription
        user_text = "User test message"
        self.app.update_transcription(user_text, is_assistant=False)
        content = self.app.transcription_text.get('1.0', tk.END)
        self.assertIn("🎤 " + user_text, content)
        
        # Test assistant transcription
        assistant_text = "Assistant test message"
        self.app.update_transcription(assistant_text, is_assistant=True)
        content = self.app.transcription_text.get('1.0', tk.END)
        self.assertIn("🤖 " + assistant_text, content)

    def test_log_message(self):
        """Test log message updates"""
        test_message = "Test log message"
        self.app.log_message(test_message)
        
        # Verify message was added to output
        log_content = self.app.output_text.get('1.0', tk.END).strip()
        self.assertEqual(log_content, test_message)
        
        # Test multiple messages
        second_message = "Second test message"
        self.app.log_message(second_message)
        log_content = self.app.output_text.get('1.0', tk.END).strip()
        self.assertIn(test_message, log_content)
        self.assertIn(second_message, log_content)

class TestClipboardManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.parent = MagicMock()
        self.parent.interface_state = {}
        self.parent.log_message = MagicMock()
        self.manager = ClipboardManager(self.parent)

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.manager.previous_content, "")
        self.assertFalse(self.manager.monitoring)
        self.assertIsNone(self.manager.monitoring_task)
        self.assertEqual(self.manager.update_interval, 0.5)
        self.assertEqual(self.manager.max_content_size, 1024 * 1024)
        self.assertEqual(self.manager.error_count, 0)
        self.assertEqual(self.manager.max_errors, 3)

    def test_detect_content_type(self):
        """Test content type detection"""
        # Test code detection
        code_samples = [
            "def test_function():",
            "class TestClass:",
            "import sys",
            "function myFunc() {",
        ]
        for code in code_samples:
            self.assertEqual(self.manager.detect_content_type(code), "code")

        # Test URL detection
        url_samples = [
            "http://example.com",
            "https://test.org",
            "www.example.com",
        ]
        for url in url_samples:
            self.assertEqual(self.manager.detect_content_type(url), "url")

        # Test text detection
        text_samples = [
            "Regular text",
            "123456",
            "No special formatting",
        ]
        for text in text_samples:
            self.assertEqual(self.manager.detect_content_type(text), "text")

    def test_looks_like_code(self):
        """Test code detection"""
        self.assertTrue(self.manager.looks_like_code("def test():"))
        self.assertTrue(self.manager.looks_like_code("class MyClass:"))
        self.assertTrue(self.manager.looks_like_code("import os"))
        self.assertFalse(self.manager.looks_like_code("regular text"))

    def test_looks_like_url(self):
        """Test URL detection"""
        self.assertTrue(self.manager.looks_like_url("http://example.com"))
        self.assertTrue(self.manager.looks_like_url("https://test.org"))
        self.assertTrue(self.manager.looks_like_url("www.example.com"))
        self.assertFalse(self.manager.looks_like_url("not a url"))

    def test_process_code(self):
        """Test code processing"""
        input_code = "def test():\n    print('test')  \n\n"
        expected = "def test():\n    print('test')\n"
        self.assertEqual(self.manager.process_code(input_code), expected)

    def test_process_text(self):
        """Test text processing"""
        input_text = "  test text  \n"
        expected = "test text"
        self.assertEqual(self.manager.process_text(input_text), expected)

    def test_process_url(self):
        """Test URL processing"""
        input_url = "  https://example.com  \n"
        expected = "https://example.com"
        self.assertEqual(self.manager.process_url(input_url), expected)

    @patch('pyperclip.paste')
    def test_get_current_content(self, mock_paste):
        """Test getting current clipboard content"""
        test_cases = [
            ("def test():\n    pass  \n\n", "def test():\n    pass\n"),  # Code
            ("  https://example.com  \n", "https://example.com"),  # URL
            ("  regular text  \n", "regular text"),  # Text
            ("", "")  # Empty
        ]
        
        for input_content, expected in test_cases:
            mock_paste.return_value = input_content
            result = self.manager.get_current_content()
            self.assertEqual(result, expected, f"Failed for input: {input_content}")

async def run_async_test(coro):
    """Helper function to run async tests"""
    try:
        return await coro
    finally:
        await asyncio.sleep(0)  # Allow other tasks to run

if __name__ == '__main__':
    unittest.main(verbosity=2)
