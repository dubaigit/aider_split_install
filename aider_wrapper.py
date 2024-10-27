import os
import argparse
import time
from queue import Queue, Empty
import json
import base64
import asyncio
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import pyaudio
from contextlib import contextmanager

# Optional imports with fallbacks
try:
    import sounddevice as sd
except ImportError:
    print("Warning: sounddevice module not found. Voice functionality will be disabled.")
    sd = None

try:
    import numpy as np
except ImportError:
    print("Warning: numpy module not found. Voice functionality will be disabled.")
    np = None

try:
    import websockets
except ImportError:
    print("Warning: websockets module not found. Voice functionality will be disabled.")
    websockets = None

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai module not found. Voice functionality will be disabled.")
    OpenAI = None

try:
    import pyperclip
except ImportError:
    print("Warning: pyperclip module not found. Clipboard functionality will be disabled.")
    pyperclip = None


# Audio settings
CHUNK_SIZE = 1024  # Smaller chunks for more responsive audio
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
REENGAGE_DELAY_MS = 500
OPENAI_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

class AudioBufferManager:
    """Manages audio buffering and processing"""
    def __init__(self, max_size, chunk_size, sample_rate):
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.stats = {'drops': 0, 'overflows': 0}
        
    def get_chunks(self, queue):
        """Get chunks from queue with overflow protection"""
        chunks = []
        while len(self.buffer) < self.max_size:
            try:
                chunk = queue.get_nowait()
                if len(self.buffer) + len(chunk) <= self.max_size:
                    chunks.append(chunk)
                else:
                    self.stats['overflows'] += 1
                    break
            except queue.Empty:
                break
        return chunks
        
    def combine_chunks(self, chunks):
        """Combine chunks with error checking"""
        try:
            return b''.join(chunks)
        except Exception as e:
            self.stats['drops'] += 1
            raise AudioProcessingError(f"Error combining chunks: {e}") from e
            
    def get_usage(self):
        """Get current buffer usage ratio"""
        return len(self.buffer) / self.max_size

class PerformanceMonitor:
    """Monitors and reports performance metrics"""
    def __init__(self, metrics, log_interval=5):
        self.metrics = {m: [] for m in metrics}
        self.last_log = time.time()
        self.log_interval = log_interval
        
    def update(self, metric, value):
        """Update metric value"""
        if metric in self.metrics:
            self.metrics[metric].append(value)
            
    def get_metrics(self):
        """Get current metric averages"""
        return {
            m: sum(v) / len(v) if v else 0 
            for m, v in self.metrics.items()
        }
        
    def should_log(self):
        """Check if it's time to log metrics"""
        if time.time() - self.last_log >= self.log_interval:
            self.last_log = time.time()
            return True
        return False
        
    def reset(self):
        """Reset all metrics"""
        self.metrics = {m: [] for m in self.metrics}

    @contextmanager
    def measure(self, metric):
        """Context manager to measure execution time of a block"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.update(metric, duration)

class KeyboardShortcuts:
    """Manages keyboard shortcuts"""
    def __init__(self, parent):
        self.parent = parent
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        shortcuts = {
            '<Control-r>': self.parent.check_all_issues,
            '<Control-a>': self.parent.browse_files,
            '<Control-v>': self.parent.use_clipboard_content,
            '<Control-s>': self.parent.send_input_text,
            '<Escape>': self.parent.stop_voice_control
        }
        
        for key, func in shortcuts.items():
            self.parent.root.bind(key, lambda e, f=func: f())

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

class ResultProcessor:
    """Processes and manages results from various operations"""
    def __init__(self, parent):
        self.parent = parent
        self.results = []
        
    def process_result(self, result, source):
        """Process a result from any operation"""
        self.results.append({
            'timestamp': time.time(),
            'result': result,
            'source': source
        })
        return result

class ErrorProcessor:
    """Processes and manages errors from various operations"""
    def __init__(self, parent):
        self.parent = parent
        self.errors = []
        
    def process_error(self, error, source):
        """Process an error from any operation"""
        error_entry = {
            'timestamp': time.time(),
            'error': str(error),
            'source': source
        }
        self.errors.append(error_entry)
        self.parent.log_message(f"Error in {source}: {error}")
        return error_entry

class VoiceCommandProcessor:
    """Processes and manages voice commands"""
    def __init__(self, parent):
        self.parent = parent
        self.commands = []
        
    def preprocess_command(self, command):
        """Clean and normalize voice command"""
        return command.strip().lower()
        
    def validate_command(self, command):
        """Validate voice command format and content"""
        if not command:
            return False
        return True

class AiderVoiceGUI:
    def __init__(self, root):
        """Initialize the AiderVoiceGUI with all required attributes."""
        self.root = root
        self.root.title("Aider Voice Assistant")
        
        # Initialize all attributes
        self.audio_buffer = bytearray()
        self.mic_stream = None
        self.spkr_stream = None
        self.response_active = False
        self.last_transcript_id = None
        self.last_audio_time = time.time()
        self.recording = False
        self.auto_mode = False
        self.audio_queue = Queue()
        self.ws = None
        self.running = True
        self.client = OpenAI() if OpenAI else None
        self.aider_process = None
        self.temp_files = []
        self.fixing_issues = False
        self.mic_active = False
        self.mic_on_at = 0
        self._stop_event = threading.Event()
        self.log_frequency = 50
        self.log_counter = 0
        self.chunk_buffer = []
        self.chunk_buffer_size = 5
        self.audio_thread = None
        
        # Core state
        self.interface_state = {
            'files': {},
            'issues': [],
            'aider_output': [],
            'clipboard_history': [],
            'last_analysis': None,
            'command_history': []
        }
        
        # Initialize managers
        self.clipboard_manager = ClipboardManager(self)
        self.result_processor = ResultProcessor(self)
        self.error_processor = ErrorProcessor(self)
        self.ws_manager = WebSocketManager(self)
        self.performance_monitor = PerformanceMonitor(['cpu', 'memory', 'latency'])
        self.keyboard_shortcuts = KeyboardShortcuts(self)
        
        # Initialize GUI components
        self.setup_gui()
        
        # Initialize asyncio loop
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()
        
        # Initialize audio components
        self.p = pyaudio.PyAudio()
        
        # Automatically start voice control
        self.start_voice_control()


    async def _send_audio_chunk(self, chunk):
        """Send audio chunk to websocket"""
        if self.ws and chunk:
            try:
                await self.ws.send(json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(chunk).decode('utf-8')
                }))
            except Exception as e:
                self.log_message(f"Error sending audio chunk: {e}")
        self.error_processor = ErrorProcessor(self)
        self.ws_manager = WebSocketManager(self)
        
        # Core attributes
        self.response_active = False
        self.last_transcript_id = None
        self.last_audio_time = time.time()
        self.recording = False
        self.auto_mode = False
        self.audio_queue = Queue()
        self.ws = None
        self.running = True
        self.client = OpenAI() if OpenAI else None
        self.aider_process = None
        self.temp_files = []
        self.fixing_issues = False
        self.mic_active = False
        self.mic_on_at = 0
        self._stop_event = threading.Event()
        
        # Performance monitoring
        self.log_frequency = 50
        self.log_counter = 0
        self.chunk_buffer = []
        self.chunk_buffer_size = 5
        self.audio_thread = None
        
        # Interface state tracking
        self.interface_state = {
            'files': {},
            'issues': [],
            'aider_output': [],
            'clipboard_history': [],
            'last_analysis': None,
            'command_history': []
        }
        
        # Parse command line arguments
        self.args = self._parse_arguments()
        
    def _parse_arguments(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Voice-controlled Aider wrapper")
        parser.add_argument("--voice-only", action="store_true", help="Run in voice control mode only")
        parser.add_argument("-i", "--instructions", help="File containing instructions")
        parser.add_argument("-c", "--clipboard", action="store_true", help="Use clipboard content as instructions")
        parser.add_argument("filenames", nargs='*', help="Filenames to process")
        parser.add_argument("--chat-mode", default="code", choices=["code", "ask"], help="Chat mode to use for aider")
        parser.add_argument("--suggest-shell-commands", action="store_true", help="Suggest shell commands while running aider")
        parser.add_argument("--model", help="Model to use for aider")
        parser.add_argument("--gui", action="store_true", help="Launch the GUI interface")
        parser.add_argument("--auto", action="store_true", help="Automatically send ruff issues to aider (GUI mode only)")
        return parser.parse_args()
        
        # Initialize managers
        self.ws_manager = WebSocketManager(self)
        self.performance_monitor = PerformanceMonitor(['cpu', 'memory', 'latency'])
        self.keyboard_shortcuts = KeyboardShortcuts(self)
        self.root.geometry("1200x800")
        
        # Initialize all attributes
        self.response_active = False
        self.last_transcript_id = None
        self.last_audio_time = time.time()
        self.recording = False
        self.auto_mode = False
        self.audio_queue = Queue()
        self.ws = None
        self.running = True
        self.client = OpenAI()
        self.aider_process = None
        self.temp_files = []
        self.fixing_issues = False
        self.mic_active = False
        self.mic_on_at = 0
        self._stop_event = threading.Event()
        self.log_frequency = 50
        self.log_counter = 0
        self.chunk_buffer = []
        self.chunk_buffer_size = 5
        self.audio_thread = None
        
        # Track interface state and content
        self.interface_state = {
            'files': {},  # Store file contents
            'issues': [],  # Store detected issues
            'aider_output': [],  # Store Aider responses
            'clipboard_history': [],  # Track clipboard content
            'last_analysis': None,  # Store last analysis results
        }
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create left panel for controls and input
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Control buttons frame
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Controls", padding="5")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Remove Voice Control button (Voice starts automatically)
        # self.voice_button = ttk.Button(
        #     self.control_frame,
        #     text="🎤 Start Voice Control",
        #     command=self.toggle_voice_control
        # )
        # self.voice_button.grid(row=0, column=0, pady=5, padx=5, sticky='ew')
        
        # Status label
        self.status_label = ttk.Label(self.control_frame, text="Initializing Voice Control...")
        self.status_label.grid(row=0, column=1, pady=5, padx=5)
        
        # Action buttons
        self.add_files_button = ttk.Button(
            self.control_frame,
            text="📁 Add Files",
            command=self.browse_files
        )
        self.add_files_button.grid(row=1, column=0, pady=5, padx=5, sticky='ew')
        
        self.check_issues_button = ttk.Button(
            self.control_frame,
            text="🔍 Check Issues",
            command=self.check_all_issues
        )
        self.check_issues_button.grid(row=1, column=1, pady=5, padx=5, sticky='ew')
        
        # Listbox to display added files
        self.files_frame = ttk.LabelFrame(self.left_panel, text="Added Files", padding="5")
        self.files_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.files_listbox = tk.Listbox(self.files_frame, height=10)
        self.files_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.remove_file_button = ttk.Button(
            self.files_frame,
            text="🗑️ Remove Selected",
            command=self.remove_selected_file
        )
        self.remove_file_button.grid(row=1, column=0, pady=5, padx=5, sticky='ew')
        
        # Input frame
        self.input_frame = ttk.LabelFrame(self.left_panel, text="Input/Instructions", padding="5")
        self.input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.input_text = scrolledtext.ScrolledText(self.input_frame, height=10)
        self.input_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.clipboard_button = ttk.Button(
            self.input_frame,
            text="📋 Load Clipboard",
            command=self.use_clipboard_content
        )
        self.clipboard_button.grid(row=1, column=0, pady=5, padx=5)
        
        self.send_button = ttk.Button(
            self.input_frame,
            text="📤 Send to Aider",
            command=self.send_input_text
        )
        self.send_button.grid(row=1, column=1, pady=5, padx=5)
        
        # Create right panel for output
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Transcription frame
        self.transcription_frame = ttk.LabelFrame(self.right_panel, text="Conversation", padding="5")
        self.transcription_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.transcription_text = scrolledtext.ScrolledText(self.transcription_frame, height=15)
        self.transcription_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Issues frame
        self.issues_frame = ttk.LabelFrame(self.right_panel, text="Issues", padding="5")
        self.issues_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.issues_text = scrolledtext.ScrolledText(self.issues_frame, height=15)
        self.issues_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create log frame
        self.log_frame = ttk.LabelFrame(self.left_panel, text="Log", padding="5")
        self.log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create output text area (for logging)
        self.output_text = scrolledtext.ScrolledText(self.log_frame, height=10)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)  # Right panel takes more space
        self.main_frame.rowconfigure(0, weight=1)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.rowconfigure(4, weight=1)  # Log frame takes remaining space
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=1)
        self.right_panel.rowconfigure(1, weight=1)
        self.files_frame.columnconfigure(0, weight=1)
        self.input_frame.columnconfigure(0, weight=1)
        self.input_frame.columnconfigure(1, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
        
        # Initialize other attributes
        self.recording = False
        self.auto_mode = False
        self.audio_queue = Queue()
        self.ws = None
        self.running = True
        self.client = OpenAI()
        self.aider_process = None
        self.temp_files = []
        self.fixing_issues = False
        
    
    def run_async_loop(self):
        """Run asyncio event loop in a separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def toggle_voice_control(self):
        """Toggle voice control on/off"""
        if not self.recording:
            self.start_voice_control()
        else:
            self.stop_voice_control()
    
    def start_voice_control(self):
        """Start voice control"""
        if None in (sd, np, websockets, OpenAI):
            self.log_message("Error: Required modules for voice control are not available")
            return
            
        self.recording = True
        # self.voice_button.configure(text="🔴 Stop Voice Control")
        self.status_label.configure(text="Listening...")
        
        # Start audio streams
        self.mic_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            stream_callback=self._mic_callback,
            frames_per_buffer=CHUNK_SIZE
        )
        self.spkr_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            stream_callback=self._spkr_callback,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        self.audio_thread.start()
        
        self.mic_stream.start_stream()
        self.spkr_stream.start_stream()
        
        # Connect to OpenAI WebSocket
        asyncio.run_coroutine_threadsafe(self.connect_websocket(), self.loop)
    
    def stop_voice_control(self):
        """Stop voice control"""
        self.recording = False
        # self.voice_button.configure(text="🎤 Start Voice Control")
        self.status_label.configure(text="Ready")
        
        # Stop audio streams
        if hasattr(self, 'mic_stream'):
            self.mic_stream.stop_stream()
            self.mic_stream.close()
        if hasattr(self, 'spkr_stream'):
            self.spkr_stream.stop_stream()
            self.spkr_stream.close()
        
        # Close WebSocket connection
        if self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            self.ws = None
        
        # Terminate PyAudio
        self.p.terminate()
    
    async def _mic_callback(self, in_data: bytes, _frame_count: int, _time_info: dict, _status: int) -> tuple[None, int]:
        """Handle microphone input callback from PyAudio.
        
        Args:
            in_data: Raw audio input data
            frame_count: Number of frames (unused but required by PyAudio API)
            time_info: Timing information (unused but required by PyAudio API)
            status: Status flags (unused but required by PyAudio API)
            
        Returns:
            tuple: (None, paContinue) as required by PyAudio API
        """
        if time.time() > self.mic_on_at:
            if not self.mic_active:
                self.log_message('🎙️🟢 Mic active')
                self.mic_active = True
            self.mic_queue.put(in_data)
            
            # Only log occasionally to reduce GUI updates
            self.log_counter += 1
            if self.log_counter % self.log_frequency == 0:
                self.log_message(f'🎤 Processing audio...')
        else:
            if self.mic_active:
                self.log_message('🎙️🔴 Mic suppressed')
                self.mic_active = False
        return (None, pyaudio.paContinue)
    
    async def _process_audio_thread(self):
        """Process audio in a separate thread with enhanced buffering and monitoring"""
        buffer_manager = AudioBufferManager(
            max_size=1024 * 1024,  # 1MB max buffer
            chunk_size=self.chunk_buffer_size,
            sample_rate=SAMPLE_RATE
        )
        
        performance_monitor = PerformanceMonitor(
            metrics=['latency', 'buffer_usage', 'processing_time']
        )
        
        while self.recording:
            try:
                with performance_monitor.measure('processing_time'):
                    # Process chunks with buffer management
                    chunks = buffer_manager.get_chunks(self.mic_queue)
                    
                    if chunks:
                        # Monitor buffer usage
                        buffer_usage = buffer_manager.get_usage()
                        performance_monitor.update('buffer_usage', buffer_usage)
                        
                        if buffer_usage > 0.8:  # 80% buffer usage warning
                            self.log_message("⚠️ High audio buffer usage")
                        
                        # Process and send audio with latency monitoring
                        start_time = time.time()
                        combined_chunk = buffer_manager.combine_chunks(chunks)
                        
                        if self.ws and self.ws_manager.connection_state == "connected":
                            await self._send_audio_chunk(combined_chunk)
                            
                        # Monitor latency
                        latency = (time.time() - start_time) * 1000
                        performance_monitor.update('latency', latency)
                        
                        if latency > 100:  # Warning for high latency
                            self.log_message(f"⚠️ High audio latency: {latency:.1f}ms")
                
                # Log performance metrics periodically
                if performance_monitor.should_log():
                    metrics = performance_monitor.get_metrics()
                    self.log_message(
                        f"📊 Audio metrics - "
                        f"Latency: {metrics['latency']:.1f}ms, "
                        f"Buffer: {metrics['buffer_usage']:.1%}, "
                        f"Processing: {metrics['processing_time']:.1f}ms"
                    )
                    
            except Exception as e:
                self.log_message(f"Error in audio processing: {e}")
                time.sleep(1)  # Delay on error
                
            await asyncio.sleep(0.01)  # Cooperative yield
    
    async def _spkr_callback(self, _in_data: bytes, frame_count: int, _time_info: dict, _status: int) -> tuple[bytes, int]:
        """Handle speaker output callback from PyAudio.
        
        Args:
            in_data: Unused input data (required by PyAudio API)
            frame_count: Number of frames to output
            time_info: Timing information (unused but required by PyAudio API)
            status: Status flags (unused but required by PyAudio API)
            
        Returns:
            tuple: (audio_data, paContinue) as required by PyAudio API
        """
        bytes_needed = frame_count * 2
        current_buffer_size = len(self.audio_buffer)

        if current_buffer_size >= bytes_needed:
            audio_chunk = bytes(self.audio_buffer[:bytes_needed])
            self.audio_buffer = bytearray(self.audio_buffer[bytes_needed:])  # Convert slice to bytearray
            self.mic_on_at = time.time() + REENGAGE_DELAY_MS / 1000
        else:
            audio_chunk = bytes(self.audio_buffer) + b'\x00' * (bytes_needed - current_buffer_size)
            self.audio_buffer = bytearray()  # Reset with empty bytearray

        return (audio_chunk, pyaudio.paContinue)
    
    async def connect_websocket(self, max_retries=3, retry_delay=2):
        """Connect to OpenAI's realtime websocket API with retry logic"""
        retries = 0
        while retries < max_retries:
            try:
                self.log_message(f"Connecting to OpenAI API (attempt {retries + 1}/{max_retries})...")
                self.ws = await websockets.connect(
                    OPENAI_WEBSOCKET_URL,
                    extra_headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                        "Content-Type": "application/json",
                        "OpenAI-Beta": "realtime=v1"
                    }
                )
                
                # Initialize session with correct configuration
                await self.ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "model": "gpt-4o",
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",  # Required parameter
                            "threshold": 0.5,
                            "prefix_padding_ms": 200,
                            "silence_duration_ms": 300
                        }
                    }
                }))
                
                # Initialize response state
                self.response_active = False
                self.last_transcript_id = None
                self.audio_buffer = bytearray()
                self.last_audio_time = time.time()
                
                # Start message handling
                asyncio.create_task(self.handle_websocket_messages())
                asyncio.create_task(self.process_audio_queue())
                
                self.log_message("Connected to OpenAI realtime API")
                return
                
            except Exception as e:
                self.log_message(f"Failed to connect to OpenAI: {e}")
                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    raise
        
        self.log_message("Failed to connect after all retries")
        self.stop_voice_control()
    
    async def process_audio_queue(self):
        """Process audio queue and send to OpenAI"""
        while self.recording:
            try:
                mic_chunk = self.mic_queue.get_nowait()
                self.log_message(f'🎤 Processing {len(mic_chunk)} bytes of audio data.')
                
                try:
                    # Send audio data to OpenAI
                    await self.ws.send(json.dumps({
                        'type': 'input_audio_buffer.append',
                        'audio': base64.b64encode(mic_chunk).decode('utf-8')
                    }))
                except Exception as e:
                    self.log_message(f"Error sending audio data: {e}")
                    
                await asyncio.sleep(0.05)
            except Empty:
                await asyncio.sleep(0.05)
            except Exception as e:
                self.log_message(f"Error processing audio queue: {e}")
                await asyncio.sleep(0.05)
    
    async def handle_websocket_messages(self):
        """Handle incoming websocket messages, including function calls."""
        while self.ws and self.recording:
            try:
                message = await self.ws.recv()
                event = json.loads(message)
                
                event_type = event.get("type")
                
                if event_type == "response.function_call":
                    function_call = event.get("delta", {}).get("function_call", {})
                    function_name = function_call.get("name")
                    arguments = json.loads(function_call.get("arguments", "{}"))

                    # Execute the local function
                    if function_name and hasattr(self, function_name):
                        self.log_message(f"Executing function: {function_name}")
                        function = getattr(self, function_name)
                        try:
                            if asyncio.iscoroutinefunction(function):
                                result = await function(**arguments)
                            else:
                                result = function(**arguments)

                            # Send the result back to the assistant
                            await self.ws.send(json.dumps({
                                "type": "function_call.result",
                                "name": function_name,
                                "result": result
                            }))
                        except Exception as e:
                            self.log_message(f"Error executing function {function_name}: {e}")
                            await self.ws.send(json.dumps({
                                "type": "function_call.error",
                                "name": function_name,
                                "error": str(e)
                            }))
                    else:
                        self.log_message(f"Unknown function called: {function_name}")
                
                elif event_type == "response.text.delta":
                    text = event.get("delta", {}).get("text", "")
                    if text.strip():
                        self.update_transcription(text, is_assistant=True)
                
                elif event_type == "input_speech_transcription_completed":
                    text = event.get("transcription", {}).get("text", "")
                    self.update_transcription(text, is_assistant=False)
                    await self.process_voice_command(text)
                
                elif event_type == "response.audio.delta":
                    try:
                        audio_content = base64.b64decode(event.get('delta', ''))
                        if audio_content:
                            self.audio_buffer.extend(audio_content)
                            self.log_message(f'Received {len(audio_content)} bytes of audio data')
                    except Exception as e:
                        self.log_message(f"Error processing audio response: {e}")
                
                elif event_type == "response.audio.done":
                    self.log_message("AI finished speaking")
                    self.response_active = False
                    
                elif event_type == "response.done":
                    self.response_active = False
                    status = event.get("status")
                    if status == "incomplete":
                        reason = event.get("status_details", {}).get("reason", "unknown")
                        self.log_message(f"🚫 Response incomplete: {reason}")
                    elif status == "failed":
                        error = event.get("status_details", {}).get("error", {})
                        self.log_message(f"⚠️ Response failed: {error.get('code', 'unknown error')}")
                    else:
                        self.log_message("Response completed")
                
                elif event_type == "error":
                    error_msg = event.get('error', {}).get('message', 'Unknown error')
                    self.log_message(f"Error from OpenAI: {error_msg}")
                    if "active response" in error_msg.lower():
                        self.response_active = True

            except websockets.exceptions.ConnectionClosed:
                self.log_message("WebSocket connection closed")
                break
            except json.JSONDecodeError as e:
                self.log_message(f"Error decoding message: {e}")
                continue
            except Exception as e:
                self.log_message(f"Error handling websocket message: {e}")
                await asyncio.sleep(1)
                continue
    

    async def process_voice_command(self, text):
        """Process transcribed voice commands with enhanced handling"""
        command_processor = VoiceCommandProcessor(self)
        
        try:
            # Update command history
            self.interface_state.setdefault('command_history', []).append({
                'timestamp': time.time(),
                'command': text,
                'status': 'processing'
            })
            
            # Pre-process command
            processed_command = command_processor.preprocess_command(text)
            self.log_message(f"Processing command: {processed_command}")
            
            # Validate command
            if not command_processor.validate_command(processed_command):
                raise ValueError("Invalid command format")
                
            # Execute command with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.ws and self.ws_manager.connection_state == "connected":
                        await self.ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{
                                    "type": "text",
                                    "text": processed_command
                                }]
                            }
                        }))
                        
                        # Update command status
                        self.interface_state['command_history'][-1]['status'] = 'completed'
                        break
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.log_message(f"Retry {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.log_message(f"❌ Command processing error: {e}")
            self.interface_state['command_history'][-1]['status'] = 'failed'
            self.interface_state['command_history'][-1]['error'] = str(e)

class ClipboardManager:
    """Manages clipboard monitoring and content processing"""
    def __init__(self, parent):
        self.parent = parent
        self.previous_content = ""
        self.monitoring = False
        self.monitoring_task = None
        self.update_interval = 0.5  # seconds
        self.max_content_size = 1024 * 1024  # 1MB
        self.history = []
        self.interface_state = parent.interface_state
        self.log_message = parent.log_message
        self.processors = {
            "code": self.process_code,
            "text": self.process_text,
            "url": self.process_url
        }
        
        # Error tracking
        self.error_count = 0
        self.max_errors = 3
        self.last_error_time = 0
        self.error_cooldown = 60  # seconds
        
    def get_current_content(self):
        """Get and process current clipboard content"""
        content = pyperclip.paste()
        content_type = self.detect_content_type(content)
        return self.processors[content_type](content)
        
    def detect_content_type(self, content):
        """Detect the type of clipboard content"""
        if self.looks_like_code(content):
            return "code"
        elif self.looks_like_url(content):
            return "url"
        return "text"
        
    def looks_like_code(self, content):
        """Check if content appears to be code"""
        code_indicators = ['def ', 'class ', 'import ', 'function', '{', '}', ';']
        return any(indicator in content for indicator in code_indicators)
        
    def looks_like_url(self, content):
        """Check if content appears to be a URL"""
        return content.startswith(('http://', 'https://', 'www.'))
        
    async def monitor_clipboard(self):
        """Monitor clipboard for changes"""
        self.monitoring = True
        while self.monitoring:
            current_content = pyperclip.paste()
            if current_content != self.previous_content:
                content_type = self.detect_content_type(current_content)
                await self.processors[content_type](current_content)
                self.previous_content = current_content
            await asyncio.sleep(0.5)
            
    def process_code(self, content):
        """Process code content"""
        # Remove unnecessary whitespace while preserving indentation
        lines = content.splitlines()
        cleaned_lines = [line.rstrip() for line in lines]
        return '\n'.join(cleaned_lines)
        
    def process_text(self, content):
        """Process text content"""
        # Basic text cleanup
        return content.strip()
        
    def process_url(self, content):
        """Process URL content"""
        # Basic URL validation and cleanup
        return content.strip()

class WebSocketManager:
    """Manages WebSocket connection state and monitoring"""
    def __init__(self, parent):
        self.parent = parent
        self.connection_state = "disconnected"
        self.last_ping_time = 0
        self.ping_interval = 30  # seconds
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.monitoring_task = None
        self.log_message = parent.log_message
        self.ws = parent.ws
        
    async def start_monitoring(self):
        """Start connection monitoring"""
        self.monitoring_task = asyncio.create_task(self._monitor_connection())
        
    async def _monitor_connection(self):
        """Monitor connection health and handle reconnection"""
        while True:
            try:
                if self.connection_state == "connected":
                    if time.time() - self.last_ping_time > self.ping_interval:
                        await self._check_connection()
                elif self.connection_state == "disconnected":
                    await self._attempt_reconnect()
                    
                await asyncio.sleep(1)
            except Exception as e:
                self.parent.log_message(f"Connection monitoring error: {e}")
                
    async def _check_connection(self):
        """Check connection health with ping"""
        try:
            if self.parent.ws:
                await self.parent.ws.ping()
                self.last_ping_time = time.time()
        except Exception:
            self.connection_state = "disconnected"
            self.parent.log_message("⚠️ WebSocket connection lost")
            await self._attempt_reconnect()
            
    async def _attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.parent.log_message("❌ Max reconnection attempts reached")
            return
            
        delay = min(30, 2 ** self.reconnect_attempts)
        self.parent.log_message(f"Attempting reconnection in {delay} seconds...")
        await asyncio.sleep(delay)
        
        try:
            await self.parent.connect_websocket()
            self.connection_state = "connected"
            self.reconnect_attempts = 0
            self.parent.log_message("✅ Successfully reconnected")
        except Exception as e:
            self.reconnect_attempts += 1
            self.parent.log_message(f"Reconnection attempt failed: {e}")
    def setup_gui(self):
        """Setup GUI components"""
        self.root.geometry("1200x800")
        
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create left panel for controls and input
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Control buttons frame
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Controls", padding="5")
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.control_frame, text="Initializing Voice Control...")
        self.status_label.grid(row=0, column=1, pady=5, padx=5)
        
        # Action buttons
        self.add_files_button = ttk.Button(
            self.control_frame,
            text="📁 Add Files",
            command=self.browse_files
        )
        self.add_files_button.grid(row=1, column=0, pady=5, padx=5, sticky='ew')
        
        self.check_issues_button = ttk.Button(
            self.control_frame,
            text="🔍 Check Issues",
            command=self.check_all_issues
        )
        self.check_issues_button.grid(row=1, column=1, pady=5, padx=5, sticky='ew')
        
        # Listbox to display added files
        self.files_frame = ttk.LabelFrame(self.left_panel, text="Added Files", padding="5")
        self.files_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.files_listbox = tk.Listbox(self.files_frame, height=10)
        self.files_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.remove_file_button = ttk.Button(
            self.files_frame,
            text="🗑️ Remove Selected",
            command=self.remove_selected_file
        )
        self.remove_file_button.grid(row=1, column=0, pady=5, padx=5, sticky='ew')
        
        # Input frame
        self.input_frame = ttk.LabelFrame(self.left_panel, text="Input/Instructions", padding="5")
        self.input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.input_text = scrolledtext.ScrolledText(self.input_frame, height=10)
        self.input_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.clipboard_button = ttk.Button(
            self.input_frame,
            text="📋 Load Clipboard",
            command=self.use_clipboard_content
        )
        self.clipboard_button.grid(row=1, column=0, pady=5, padx=5)
        
        self.send_button = ttk.Button(
            self.input_frame,
            text="📤 Send to Aider",
            command=self.send_input_text
        )
        self.send_button.grid(row=1, column=1, pady=5, padx=5)
        
        # Create right panel for output
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Transcription frame
        self.transcription_frame = ttk.LabelFrame(self.right_panel, text="Conversation", padding="5")
        self.transcription_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.transcription_text = scrolledtext.ScrolledText(self.transcription_frame, height=15)
        self.transcription_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Issues frame
        self.issues_frame = ttk.LabelFrame(self.right_panel, text="Issues", padding="5")
        self.issues_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.issues_text = scrolledtext.ScrolledText(self.issues_frame, height=15)
        self.issues_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create log frame
        self.log_frame = ttk.LabelFrame(self.left_panel, text="Log", padding="5")
        self.log_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create output text area (for logging)
        self.output_text = scrolledtext.ScrolledText(self.log_frame, height=10)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=2)  # Right panel takes more space
        self.main_frame.rowconfigure(0, weight=1)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.rowconfigure(4, weight=1)  # Log frame takes remaining space
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.rowconfigure(0, weight=1)
        self.right_panel.rowconfigure(1, weight=1)
        self.files_frame.columnconfigure(0, weight=1)
        self.input_frame.columnconfigure(0, weight=1)
        self.input_frame.columnconfigure(1, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
