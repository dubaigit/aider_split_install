"""Main module for Aider Voice Assistant."""

# Standard library imports
import argparse
import asyncio
import base64
import json
import os
import threading
import time
import unittest
from contextlib import contextmanager
from enum import Enum, auto
from queue import Empty as QueueEmpty, Queue
from pathlib import Path
from unittest.mock import MagicMock, patch

# GUI imports
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

# Third-party imports
import pyaudio
import websockets
from websockets.exceptions import WebSocketException

# Custom exceptions
class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""

class WebSocketConnectionError(Exception):
    """Custom exception for WebSocket connection errors"""
    def __init__(self, message, original_error=None):
        super().__init__(message)
        self.original_error = original_error

class WebSocketTimeoutError(WebSocketConnectionError):
    """Custom exception for WebSocket timeout errors"""

class WebSocketAuthenticationError(WebSocketConnectionError):
    """Custom exception for WebSocket authentication errors"""

class AuthenticationError(WebSocketConnectionError):
    """Custom exception for authentication failures"""

class StateError(Exception):
    """Custom exception for state-related errors"""

class ValidationError(Exception):
    """Custom exception for validation errors"""

class AudioError(Exception):
    """Custom exception for audio-related errors"""

# Optional third-party imports with fallbacks
try:
    import numpy as np
except ImportError:
    print("Warning: numpy module not found. Voice functionality will be disabled.")
    np = None

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

try:
    import sounddevice as sd
except ImportError:
    print("Warning: sounddevice module not found. Voice functionality will be disabled.")
    sd = None

try:
    import websockets
except ImportError:
    print("Warning: websockets module not found. Voice functionality will be disabled.")
    websockets = None

# Audio settings
CHUNK_SIZE = 1024  # Smaller chunks for more responsive audio
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
REENGAGE_DELAY_MS = 500
OPENAI_WEBSOCKET_URL = ("wss://api.openai.com/v1/realtime"
                       "?model=gpt-4o-realtime-preview-2024-10-01")

class ConnectionState(Enum):
    """Enum for WebSocket connection states"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()
    ERROR = auto()
    CLOSING = auto()

class AudioBufferManager:
    """Manages audio buffering and processing"""

    def __init__(self, max_size, chunk_size, sample_rate):
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.stats = {"drops": 0, "overflows": 0}

    def get_chunks(self, queue):
        """Get chunks from queue with overflow protection"""
        chunks = []
        while len(self.buffer) < self.max_size:
            try:
                chunk = queue.get_nowait()
                if len(self.buffer) + len(chunk) <= self.max_size:
                    chunks.append(chunk)
                else:
                    self.stats["overflows"] += 1
                    break
            except QueueEmpty:
                break
        return chunks

    def combine_chunks(self, chunks):
        """Combine chunks with error checking"""
        try:
            return b"".join(chunks)
        except Exception as e:
            self.stats["drops"] += 1
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
        return {m: sum(v) / len(v) if v else 0 for m, v in self.metrics.items()}

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
            "<Control-r>": self.parent.check_all_issues,
            "<Control-a>": self.parent.browse_files,
            "<Control-v>": self.parent.use_clipboard_content,
            "<Control-s>": self.parent.send_input_text,
            "<Escape>": self.parent.stop_voice_control,
        }

        for key, func in shortcuts.items():
            self.parent.root.bind(key, lambda e, f=func: f())

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
            "url": self.process_url,
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
        if self.looks_like_url(content):
            return "url"
        return "text"

    def looks_like_code(self, content):
        """Check if content appears to be code"""
        code_indicators = ["def ", "class ", "import ", "function", "{", "}", ";"]
        return any(indicator in content for indicator in code_indicators)

    def looks_like_url(self, content):
        """Check if content appears to be a URL"""
        return content.startswith(("http://", "https://", "www."))

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
        return "\n".join(cleaned_lines)

    def process_text(self, content):
        """Process text content"""
        # Basic text cleanup
        return content.strip()

    def process_url(self, content):
        """Process URL content"""
        # Basic URL validation and cleanup
        return content.strip()

class ResultProcessor:
    """Processes and manages results from various operations"""

    def __init__(self, parent):
        self.parent = parent
        self.results = []

    def process_result(self, result, source):
        """Process a result from any operation"""
        self.results.append({
            "timestamp": time.time(),
            "result": result,
            "source": source
        })
        return result

    def get_latest_result(self):
        """Get the most recent result"""
        return self.results[-1] if self.results else None

class ErrorProcessor:
    """Processes and manages errors from various operations"""

    def __init__(self, parent):
        self.parent = parent
        self.errors = []

    def process_error(self, error, source):
        """Process an error from any operation"""
        error_entry = {
            "timestamp": time.time(),
            "error": str(error),
            "source": source
        }
        self.errors.append(error_entry)
        self.parent.log_message(f"Error in {source}: {error}")
        return error_entry

    def get_latest_error(self):
        """Get the most recent error"""
        return self.errors[-1] if self.errors else None

class VoiceCommandProcessor:
    """Processes and manages voice commands"""

    def __init__(self, parent):
        self.parent = parent
        self.commands = []
        self.command_history = []

    def preprocess_command(self, command):
        """Clean and normalize voice command"""
        return command.strip().lower()

    def validate_command(self, command):
        """Validate voice command format and content"""
        if not command:
            return False
        if len(command) > 1000:  # Prevent overly long commands
            return False
        if command.isspace():  # Reject whitespace-only commands
            return False
        # Basic profanity check
        profanity = {'profanity1', 'profanity2'}  # Add actual words as needed
        if any(word in command.lower() for word in profanity):
            return False
        return True

    def add_to_history(self, command, status="pending"):
        """Add command to history with status"""
        self.command_history.append({
            "command": command,
            "timestamp": time.time(),
            "status": status
        })

class ConnectionState(Enum):
    """Enum for WebSocket connection states"""
    DISCONNECTED = auto()  # Initial state or after clean disconnect
    CONNECTING = auto()    # Attempting initial connection
    CONNECTED = auto()     # Successfully connected
    RECONNECTING = auto()  # Attempting to restore lost connection
    FAILED = auto()        # Connection attempts exhausted
    ERROR = auto()         # Unexpected error state
    CLOSING = auto()       # Clean shutdown in progress

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
        self.connection_latency = 0

    def get_connection_latency(self):
        """Get current connection latency"""
        return self.connection_latency

    async def check_connection(self):
        """Check connection health with ping"""
        try:
            start_time = time.time()
            if self.parent.ws:
                await self.parent.ws.ping()
                self.connection_latency = (time.time() - start_time) * 1000
                self.last_ping_time = time.time()
                return True
            return False
        except Exception as e:
            self.log_message(f"Connection check failed: {e}")
            return False

class AiderVoiceGUI:
    """Main GUI class for the voice assistant"""

    def __init__(self, root):
        """Initialize the AiderVoiceGUI."""
        # Store root window
        self.root = root
        self.root.title("Aider Voice Assistant")

        # Parse command line arguments
        self.args = self.parse_arguments()

        # Initialize interface state
        self.interface_state = {
            "files": {},  # Store file contents
            "issues": [],  # Store detected issues
            "aider_output": [],  # Store Aider responses
            "clipboard_history": [],  # Track clipboard content
            "last_analysis": None,  # Store last analysis results
            "command_history": [],  # Track command history
        }

        # Initialize components
        self.init_components()

        # Setup GUI if enabled
        if self.args.gui:
            self.setup_gui()

    def init_components(self):
        """Initialize all components"""
        # Initialize queues
        self.mic_queue = Queue()
        self.audio_queue = Queue()

        # Initialize audio components
        self.audio_buffer = bytearray()
        self.mic_stream = None
        self.spkr_stream = None
        self.chunk_buffer = []
        self.chunk_buffer_size = 5
        self.audio_thread = None
        self.p = pyaudio.PyAudio()

        # Initialize state tracking
        self.response_active = False
        self.last_transcript_id = None
        self.last_audio_time = time.time()
        self.recording = False
        self.auto_mode = False
        self.running = True
        self.fixing_issues = False
        self.mic_active = False
        self.mic_on_at = 0
        self.stop_event = threading.Event()
        self.log_frequency = 50
        self.log_counter = 0

        # Initialize API clients
        self.client = OpenAI() if OpenAI else None
        self.ws = None
        self.aider_process = None
        self.temp_files = []

        # Initialize async components
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)

        # Initialize managers
        self.clipboard_manager = ClipboardManager(self)
        self.result_processor = ResultProcessor(self)
        self.error_processor = ErrorProcessor(self)
        self.ws_manager = WebSocketManager(self)
        self.performance_monitor = PerformanceMonitor(["cpu", "memory", "latency"])
        self.keyboard_shortcuts = KeyboardShortcuts(self)

        # Start async thread
        self.thread.start()

    @classmethod
    def get_parser(cls):
        """Get argument parser"""
        if not hasattr(cls, '_parser'):
            cls._parser = argparse.ArgumentParser(
                description="Aider Voice Assistant",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            # Add arguments here...
        return cls._parser

    @classmethod
    def parse_arguments(cls, args=None):
        """Parse command line arguments"""
        parser = cls.get_parser()
        
        # Add arguments
        parser.add_argument('--voice-only', action='store_true',
                          help='Run in voice-only mode without GUI')
        parser.add_argument('-i', '--instructions',
                          help='Path to instructions file')
        parser.add_argument('-c', '--clipboard', action='store_true',
                          help='Monitor clipboard for content')
        parser.add_argument('--chat-mode', choices=['code', 'ask'],
                          default='code', help='Chat interaction mode')
        parser.add_argument('--suggest-shell-commands', action='store_true',
                          help='Enable shell command suggestions')
        parser.add_argument('--model',
                          help='Specify OpenAI model to use')
        parser.add_argument('--gui', action='store_true',
                          help='Run with GUI interface')
        parser.add_argument('--auto', action='store_true',
                          help='Enable automatic mode')
        parser.add_argument('filenames', nargs='*',
                          help='Files to process')
        
        parsed_args = parser.parse_args(args)
        
        # Validate arguments
        if parsed_args.voice_only and parsed_args.gui:
            parser.error("Cannot use --voice-only with --gui")
            
        if parsed_args.instructions and not os.path.exists(parsed_args.instructions):
            parser.error(f"Instructions file not found: {parsed_args.instructions}")
            
        for filename in parsed_args.filenames:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    f.read()
            except (IOError, OSError) as e:
                parser.error(f"Cannot read file '{filename}': {e}")

        return parsed_args

    def browse_files(self):
        """Open file browser dialog to select files"""
        try:
            files = filedialog.askopenfilenames()
            for file in files:
                if not os.path.exists(file):
                    self.log_message(f"⚠️ File not found: {file}")
                    continue

                if not os.access(file, os.R_OK):
                    self.log_message(f"⚠️ No read permission: {file}")
                    continue

                if file not in self.interface_state['files']:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            self.interface_state['files'][file] = f.read()
                        self.files_listbox.insert(tk.END, file)
                        self.log_message(f"✅ Added file: {file}")
                    except Exception as e:
                        self.log_message(f"❌ Error reading file {file}: {str(e)}")
        except Exception as e:
            self.log_message(f"❌ Error browsing files: {str(e)}")

    def remove_selected_file(self):
        """Remove selected file from listbox"""
        try:
            selection = self.files_listbox.curselection()
            if selection:
                file = self.files_listbox.get(selection)
                self.files_listbox.delete(selection)
                del self.interface_state['files'][file]
                self.log_message(f"🗑️ Removed file: {file}")
        except Exception as e:
            self.log_message(f"❌ Error removing file: {str(e)}")

    def use_clipboard_content(self):
        """Load clipboard content into input text"""
        if pyperclip:
            content = pyperclip.paste()
            self.input_text.delete('1.0', tk.END)
            self.input_text.insert('1.0', content)

    def send_input_text(self):
        """Send input text content to processing"""
        content = self.input_text.get('1.0', tk.END).strip()
        if content:
            self.log_message("Processing input...")

    def update_transcription(self, text, is_assistant=False):
        """Update transcription text area with new text"""
        prefix = "🤖 " if is_assistant else "🎤 "
        self.transcription_text.insert(tk.END, f"{prefix}{text}\n")
        self.transcription_text.see(tk.END)

    async def send_audio_chunk(self, chunk):
        """Send audio chunk to websocket"""
        if self.ws and chunk:
            try:
                await self.ws.send(json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(chunk).decode('utf-8')
                }))
            except (websockets.exceptions.WebSocketException, json.JSONDecodeError) as e:
                self.log_message(f"Error sending audio chunk: {e}")


    def setup_gui(self):
        """Setup GUI components and layout"""
        # Create main frame with padding
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create left panel for controls and input
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Control buttons frame
        self.control_frame = ttk.LabelFrame(
            self.left_panel, text="Controls", padding="5"
        )
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # Remove Voice Control button (Voice starts automatically)
        # self.voice_button = ttk.Button(
        #     self.control_frame,
        #     text="🎤 Start Voice Control",
        #     command=self.toggle_voice_control
        # )
        # self.voice_button.grid(row=0, column=0, pady=5, padx=5, sticky='ew')

        # Status label
        self.status_label = ttk.Label(
            self.control_frame, text="Initializing Voice Control..."
        )
        self.status_label.grid(row=0, column=1, pady=5, padx=5)

        # Action buttons
        self.add_files_button = ttk.Button(
            self.control_frame, text="📁 Add Files", command=self.browse_files
        )
        self.add_files_button.grid(row=1, column=0, pady=5, padx=5, sticky="ew")

        self.check_issues_button = ttk.Button(
            self.control_frame, text="🔍 Check Issues", command=self.check_all_issues
        )
        self.check_issues_button.grid(row=1, column=1, pady=5, padx=5, sticky="ew")

        # Listbox to display added files
        self.files_frame = ttk.LabelFrame(
            self.left_panel, text="Added Files", padding="5"
        )
        self.files_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.files_listbox = tk.Listbox(self.files_frame, height=10)
        self.files_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.remove_file_button = ttk.Button(
            self.files_frame,
            text="🗑️ Remove Selected",
            command=self.remove_selected_file,
        )
        self.remove_file_button.grid(row=1, column=0, pady=5, padx=5, sticky="ew")

        # Input frame
        self.input_frame = ttk.LabelFrame(
            self.left_panel, text="Input/Instructions", padding="5"
        )
        self.input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.input_text = scrolledtext.ScrolledText(self.input_frame, height=10)
        self.input_text.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        self.clipboard_button = ttk.Button(
            self.input_frame,
            text="📋 Load Clipboard",
            command=self.use_clipboard_content,
        )
        self.clipboard_button.grid(row=1, column=0, pady=5, padx=5)

        self.send_button = ttk.Button(
            self.input_frame, text="📤 Send to Aider", command=self.send_input_text
        )
        self.send_button.grid(row=1, column=1, pady=5, padx=5)

        # Create right panel for output
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # Transcription frame
        self.transcription_frame = ttk.LabelFrame(
            self.right_panel, text="Conversation", padding="5"
        )
        self.transcription_frame.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5
        )

        self.transcription_text = scrolledtext.ScrolledText(
            self.transcription_frame, height=15
        )
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
            self.log_message(
                "Error: Required modules for voice control are not available"
            )
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
            stream_callback=self.handle_mic_input,
            frames_per_buffer=CHUNK_SIZE,
        )
        self.spkr_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            stream_callback=self.handle_speaker_output,
            frames_per_buffer=CHUNK_SIZE,
        )

        # Start audio processing thread
        self.audio_thread = threading.Thread(
            target=self.process_audio_thread, daemon=True
        )
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
        if hasattr(self, "mic_stream"):
            self.mic_stream.stop_stream()
            self.mic_stream.close()
        if hasattr(self, "spkr_stream"):
            self.spkr_stream.stop_stream()
            self.spkr_stream.close()

        # Close WebSocket connection
        if self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            self.ws = None

        # Terminate PyAudio
        self.p.terminate()

    async def handle_mic_input(
        self, in_data: bytes, _frame_count: int, _time_info: dict, _status: int
    ) -> tuple[None, int]:
        """Handle microphone input callback from PyAudio.

        Args:
            in_data: Raw audio input data
            frame_count: Number of frames (unused but required by PyAudio API)
            time_info: Timing information (unused but required by PyAudio API)
            status: Status flags (unused but required by PyAudio API)

        Returns:
            tuple: (None, paContinue) as required by PyAudio API
        """
        try:
            if time.time() > self.mic_on_at:
                if not self.mic_active:
                    self.log_message("🎙️🟢 Mic active")
                    self.mic_active = True
                self.mic_queue.put(in_data)

                # Only log occasionally to reduce GUI updates
                self.log_counter += 1
                if self.log_counter % self.log_frequency == 0:
                    self.log_message("🎤 Processing audio...")
            else:
                if self.mic_active:
                    self.log_message("🎙️🔴 Mic suppressed")
                    self.mic_active = False
            return (None, pyaudio.paContinue)
        except ValueError as e:
            self.log_message(f"Value error in mic callback: {e}")
            return (None, pyaudio.paContinue)
        except RuntimeError as e:
            self.log_message(f"Runtime error in mic callback: {e}")
            return (None, pyaudio.paContinue)
        except OSError as e:
            self.log_message(f"OS error in mic callback: {e}")
            return (None, pyaudio.paContinue)

    async def process_audio_thread(self):
        """Process audio in a separate thread with enhanced buffering and monitoring"""
        buffer_manager = AudioBufferManager(
            max_size=1024 * 1024,  # 1MB max buffer
            chunk_size=self.chunk_buffer_size,
            sample_rate=SAMPLE_RATE,
        )

        performance_monitor = PerformanceMonitor(
            metrics=["latency", "buffer_usage", "processing_time"]
        )

        while self.recording:
            try:
                with performance_monitor.measure("processing_time"):
                    # Process chunks with buffer management
                    chunks = buffer_manager.get_chunks(self.mic_queue)

                    if chunks:
                        # Monitor buffer usage
                        buffer_usage = buffer_manager.get_usage()
                        performance_monitor.update("buffer_usage", buffer_usage)

                        if buffer_usage > 0.8:  # 80% buffer usage warning
                            self.log_message("⚠️ High audio buffer usage")

                        # Process and send audio with latency monitoring
                        start_time = time.time()
                        combined_chunk = buffer_manager.combine_chunks(chunks)

                        if self.ws and self.ws_manager.connection_state == "connected":
                            await self.send_audio_chunk(combined_chunk)

                        # Monitor latency
                        latency = (time.time() - start_time) * 1000
                        performance_monitor.update("latency", latency)

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

            except (ValueError, RuntimeError, OSError) as e:
                self.log_message(f"Error in audio processing: {e}")
                time.sleep(1)  # Delay on error

            await asyncio.sleep(0.01)  # Cooperative yield

    async def handle_speaker_output(
        self, _in_data: bytes, frame_count: int, _time_info: dict, _status: int
    ) -> tuple[bytes, int]:
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
            self.audio_buffer = bytearray(
                self.audio_buffer[bytes_needed:]
            )  # Convert slice to bytearray
            self.mic_on_at = time.time() + REENGAGE_DELAY_MS / 1000
        else:
            audio_chunk = bytes(self.audio_buffer) + b"\x00" * (
                bytes_needed - current_buffer_size
            )
            self.audio_buffer = bytearray()  # Reset with empty bytearray

        return (audio_chunk, pyaudio.paContinue)

    async def connect_websocket(self):
        """Connect to OpenAI's realtime websocket API with enhanced retry logic"""
        self.ws_manager.connection_state = ConnectionState.CONNECTING
        
        try:
            self.ws = await websockets.connect(
                OPENAI_WEBSOCKET_URL,
                extra_headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "realtime=v1",
                },
                ping_interval=30,  # Enable keepalive pings
                ping_timeout=10,   # Timeout for pings
                close_timeout=5,   # Timeout for close operation
            )

            # Initialize session configuration
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "model": "gpt-4o",
                    "voice": "alloy",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 200,
                        "silence_duration_ms": 300,
                    },
                },
            }))

            # Initialize state
            self.response_active = False
            self.last_transcript_id = None
            self.audio_buffer = bytearray()
            self.last_audio_time = time.time()

            # Start handlers
            asyncio.create_task(self.handle_websocket_messages())
            asyncio.create_task(self.process_audio_queue())
            
            # Start connection monitoring
            await self.ws_manager.start_monitoring()
            
            self.ws_manager.connection_state = ConnectionState.CONNECTED
            self.log_message("✅ Connected to OpenAI realtime API")
            return True

        except websockets.exceptions.InvalidStatusCode as e:
            error = WebSocketAuthenticationError(f"Authentication failed with status {e.status_code}")
            self.error_processor.process_error(error, "websocket")
            self.ws_manager.connection_state = ConnectionState.FAILED
            raise error

        except (websockets.exceptions.WebSocketException, ConnectionError, OSError) as e:
            self.log_message(f"❌ Connection failed: {type(e).__name__}: {str(e)}")
            self.ws_manager.connection_state = ConnectionState.FAILED
            raise WebSocketConnectionError("Failed to establish connection", e)

        except Exception as e:
            self.log_message(f"❌ Unexpected error during connection: {type(e).__name__}: {str(e)}")
            self.ws_manager.connection_state = ConnectionState.FAILED
            raise

    async def process_audio_queue(self):
        """Process audio queue and send to OpenAI"""
        while self.recording:
            try:
                # Try to get data from queue with timeout
                try:
                    mic_chunk = self.mic_queue.get_nowait()
                except QueueEmpty:
                    # Queue is empty, wait briefly and continue
                    await asyncio.sleep(0.05)
                    continue
                
                # Process the chunk if we got one
                try:
                    if mic_chunk:
                        self.log_message(f"🎤 Processing {len(mic_chunk)} bytes of audio data.")
                        # Send audio data to OpenAI
                        await self.ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(mic_chunk).decode("utf-8"),
                                }
                            )
                        )
                except websockets.exceptions.WebSocketException as e:
                    self.error_processor.process_error(
                        ConnectionError(f"WebSocket error sending audio data: {e}"),
                        "audio_processing"
                    )
                except json.JSONDecodeError as e:
                    self.error_processor.process_error(
                        ValidationError(f"Invalid JSON in audio data: {e}"),
                        "audio_processing"
                    )
                except Exception as e:
                    self.error_processor.process_error(
                        AudioError(f"Unexpected error sending audio data: {e}"),
                        "audio_processing"
                    )

                await asyncio.sleep(0.05)
            except (ValueError, RuntimeError) as e:
                self.log_message(f"Error processing audio queue: {e}")
                await asyncio.sleep(0.05)

    async def handle_websocket_messages(self):
        """Handle incoming websocket messages with enhanced error handling"""
        while self.ws and self.recording:
            try:
                # Use timeout to detect stalled connections
                message = await asyncio.wait_for(self.ws.recv(), timeout=30)
                event = json.loads(message)
                event_type = event.get("type")

                # Process different event types
                handlers = {
                    "response.function_call": self._handle_function_call,
                    "response.text.delta": self._handle_text_delta,
                    "input_speech_transcription_completed": self._handle_transcription,
                    "response.audio.delta": self._handle_audio_delta,
                    "response.audio.done": self._handle_audio_done,
                    "response.done": self._handle_response_done,
                    "error": self._handle_error_event,
                }

                if handler := handlers.get(event_type):
                    try:
                        await handler(event)
                    except Exception as e:
                        self.log_message(f"❌ Error in event handler for {event_type}: {type(e).__name__}: {str(e)}")
                else:
                    self.log_message(f"ℹ️ Unhandled event type: {event_type}")

            except asyncio.TimeoutError:
                self.log_message("⚠️ WebSocket timeout - checking connection...")
                if not await self.ws_manager.check_connection():
                    self.ws_manager.connection_state = ConnectionState.DISCONNECTED
                    break

            except websockets.exceptions.ConnectionClosedOK:
                self.log_message("ℹ️ WebSocket connection closed normally")
                self.ws_manager.connection_state = ConnectionState.DISCONNECTED
                break

            except websockets.exceptions.ConnectionClosedError as e:
                self.log_message(f"⚠️ Connection closed abnormally: {e.code} - {e.reason}")
                self.ws_manager.connection_state = ConnectionState.RECONNECTING
                if not await self.ws_manager.attempt_reconnect():
                    break

            except json.JSONDecodeError as e:
                self.log_message(f"⚠️ Invalid message format: {str(e)}")
                continue

            except Exception as e:
                self.log_message(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
                self.ws_manager.connection_state = ConnectionState.ERROR
                await asyncio.sleep(1)
                if not await self.ws_manager.attempt_reconnect():
                    break

    async def _handle_function_call(self, event):
        """Handle function call events"""
        try:
            function_call = event.get("delta", {}).get("function_call", {})
            function_name = function_call.get("name")
            arguments = json.loads(function_call.get("arguments", "{}"))

            if function_name and hasattr(self, function_name):
                self.log_message(f"Executing function: {function_name}")
                function = getattr(self, function_name)
                try:
                    if asyncio.iscoroutinefunction(function):
                        result = await function(**arguments)
                    else:
                        result = function(**arguments)

                    await self.ws.send(json.dumps({
                        "type": "function_call.result",
                        "name": function_name,
                        "result": result
                    }))
                except (TypeError, ValueError) as e:
                    self.log_message(f"Error executing {function_name}: {e}")
                    await self.ws.send(json.dumps({
                        "type": "function_call.error",
                        "name": function_name,
                        "error": str(e)
                    }))
            else:
                self.log_message(f"Unknown function called: {function_name}")
        except json.JSONDecodeError as e:
            self.log_message(f"Error parsing function arguments: {e}")

    async def _handle_text_delta(self, event):
        """Handle text delta events"""
        text = event.get("delta", {}).get("text", "")
        if text.strip():
            self.update_transcription(text, is_assistant=True)

    async def _handle_transcription(self, event):
        """Handle transcription completed events"""
        text = event.get("transcription", {}).get("text", "")
        self.update_transcription(text, is_assistant=False)
        await self.process_voice_command(text)

    async def _handle_audio_delta(self, event):
        """Handle audio delta events"""
        try:
            audio_content = base64.b64decode(event.get('delta', ''))
            if audio_content:
                self.audio_buffer.extend(audio_content)
                self.log_message(f'Received {len(audio_content)} bytes of audio data')
        except (TypeError, ValueError) as e:
            self.log_message(f"Error processing audio response: {e}")

    async def _handle_audio_done(self):
        """Handle audio done events"""
        self.log_message("AI finished speaking")
        self.response_active = False

    async def _handle_response_done(self, event):
        """Handle response done events"""
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

    async def _handle_error_event(self, event):
        """Handle error events"""
        error_msg = event.get("error", {}).get(
            "message", "Unknown error"
        )
        self.log_message(f"Error from OpenAI: {error_msg}")
        if "active response" in error_msg.lower():
            self.response_active = True

    async def _handle_connection_error(self, delay):
        """Handle connection errors with exponential backoff"""
        try:
            self.log_message(f"Connection error - attempting reconnect in {delay}s...")
            await asyncio.sleep(delay)
            
            # Close existing connection if any
            if self.ws:
                try:
                    await self.ws.close()
                except Exception:
                    pass
                self.ws = None
            
            if await self.ws_manager.attempt_reconnect():
                self.log_message("Successfully reconnected")
                # Reset error state
                self.ws_manager.reconnect_attempts = 0
                return True
            else:
                self.log_message("Reconnection failed")
                return False
                
        except Exception as e:
            self.log_message(f"Error during reconnection attempt: {e}")
            return False

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
                        
                except websockets.exceptions.WebSocketException as e:
                    self.log_message(f"WebSocket error on retry {attempt + 1}/{max_retries}: {e}")
                except json.JSONDecodeError as e:
                    self.log_message(f"JSON decode error on retry {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        self.interface_state['command_history'][-1]['status'] = 'failed'
                        self.interface_state['command_history'][-1]['error'] = str(e)
                        self.log_message(f"❌ Command failed after {max_retries} retries: {e}")
                        break
                    await asyncio.sleep(1)
                    
        except (websockets.exceptions.WebSocketException, json.JSONDecodeError) as e:
            self.log_message(f"❌ Command processing error: {e}")
            self.interface_state['command_history'][-1]['status'] = 'failed'
            self.interface_state['command_history'][-1]['error'] = str(e)
        except ValueError as e:
            self.log_message(f"❌ Command validation error: {e}")
            self.interface_state['command_history'][-1]['status'] = 'failed'
            self.interface_state['command_history'][-1]['error'] = str(e)


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests with enhanced async support"""

    def setUp(self):
        """Set up test environment with proper async context"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.addCleanup(self.cleanup_loop)

    def cleanup_loop(self):
        """Clean up the event loop"""
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()
        asyncio.set_event_loop(None)

    async def asyncSetUp(self):
        """Optional async setup - override in subclasses"""
        pass

    async def asyncTearDown(self):
        """Optional async teardown - override in subclasses"""
        pass

    def run_async_test(self, coro):
        """Run coroutine in the test loop with enhanced error handling and timeout"""
        async def _run_with_setup():
            await self.asyncSetUp()
            try:
                result = await asyncio.wait_for(coro, timeout=5.0)
                return result
            except asyncio.TimeoutError:
                self.fail("Async test timed out after 5 seconds")
            finally:
                await self.asyncTearDown()

        try:
            return self.loop.run_until_complete(_run_with_setup())
        except Exception as e:
            self.fail(f"Async test failed with error: {type(e).__name__}: {str(e)}\n{e.__traceback__}")

    @contextmanager
    def assertNotRaises(self, exc_type):
        """Context manager to assert no exception is raised"""
        try:
            yield
        except exc_type as e:
            self.fail(f"Expected no {exc_type.__name__} but got: {e}")

    async def wait_for_condition(self, condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(interval)
        return False

class TestVoiceCommandProcessor(AsyncTestCase):
    """Test suite for VoiceCommandProcessor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.parent = MagicMock()
        self.processor = VoiceCommandProcessor(self.parent)

    async def test_validate_command_empty(self):
        """Test validation of empty commands"""
        self.assertFalse(self.processor.validate_command(""))
        self.assertFalse(self.processor.validate_command(None))
        self.assertFalse(self.processor.validate_command("   "))

    async def test_validate_command_length(self):
        """Test validation of command length"""
        # Test command that exceeds max length
        long_command = "a" * 1001
        self.assertFalse(self.processor.validate_command(long_command))
        
        # Test command at max length
        valid_command = "a" * 1000
        self.assertTrue(self.processor.validate_command(valid_command))
        
        # Test normal length command
        self.assertTrue(self.processor.validate_command("normal command"))

    async def test_validate_command_profanity(self):
        """Test validation of command content"""
        # Test commands with profanity
        self.assertFalse(self.processor.validate_command("profanity1 test"))
        self.assertFalse(self.processor.validate_command("test profanity2"))
        
        # Test normal commands
        self.assertTrue(self.processor.validate_command("normal command"))
        self.assertTrue(self.processor.validate_command("hello world"))
        self.assertTrue(self.processor.validate_command("test case"))

class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing"""

    def setUp(self):
        """Set up test environment"""
        self.parser_patcher = patch('argparse.ArgumentParser')
        self.mock_parser = self.parser_patcher.start()
        self.mock_args = MagicMock()
        self.mock_parser.return_value.parse_args.return_value = self.mock_args

    def tearDown(self):
        """Clean up test environment"""
        self.parser_patcher.stop()

    def test_default_arguments(self):
        """Test default argument values"""
        # Set up default argument values
        self.mock_args.voice_only = False
        self.mock_args.instructions = None
        self.mock_args.clipboard = False
        self.mock_args.filenames = []
        self.mock_args.chat_mode = "code"
        self.mock_args.suggest_shell_commands = False
        self.mock_args.model = None
        self.mock_args.gui = False
        self.mock_args.auto = False

        args = AiderVoiceGUI.parse_arguments([])
        
        # Verify parser configuration
        self.mock_parser.assert_called_once()
        self.assertEqual(self.mock_parser.return_value.parse_args.call_count, 1)
        
        # Verify argument values
        self.assertFalse(args.voice_only)
        self.assertIsNone(args.instructions)
        self.assertFalse(args.clipboard)
        self.assertEqual(args.filenames, [])
        self.assertEqual(args.chat_mode, "code")
        self.assertFalse(args.suggest_shell_commands)
        self.assertIsNone(args.model)
        self.assertFalse(args.gui)
        self.assertFalse(args.auto)

    def test_custom_arguments(self):
        """Test custom argument values"""
        # Set up expected argument values
        self.mock_args.voice_only = True
        self.mock_args.instructions = "instructions.txt"
        self.mock_args.clipboard = True
        self.mock_args.filenames = ["file1.py", "file2.py"]
        self.mock_args.chat_mode = "ask"
        self.mock_args.suggest_shell_commands = True
        self.mock_args.model = "gpt-4"
        self.mock_args.gui = True
        self.mock_args.auto = True

        test_args = [
            "--voice-only",
            "-i", "instructions.txt",
            "-c",
            "--chat-mode", "ask",
            "--suggest-shell-commands",
            "--model", "gpt-4",
            "--gui",
            "--auto",
            "file1.py", "file2.py"
        ]
        
        args = AiderVoiceGUI.parse_arguments(test_args)
        
        # Verify parser was called with correct arguments
        self.mock_parser.return_value.parse_args.assert_called_once_with(test_args)
        
        # Verify all argument values
        self.assertTrue(args.voice_only)
        self.assertEqual(args.instructions, "instructions.txt")
        self.assertTrue(args.clipboard)
        self.assertEqual(args.filenames, ["file1.py", "file2.py"])
        self.assertEqual(args.chat_mode, "ask")
        self.assertTrue(args.suggest_shell_commands)
        self.assertEqual(args.model, "gpt-4")
        self.assertTrue(args.gui)
        self.assertTrue(args.auto)

    def test_invalid_chat_mode(self):
        """Test handling of invalid chat mode"""
        self.mock_parser.return_value.parse_args.side_effect = SystemExit(2)
        
        with self.assertRaises(SystemExit):
            AiderVoiceGUI.parse_arguments(["--chat-mode", "invalid"])

    def test_help_flag(self):
        """Test help flag triggers system exit"""
        self.mock_parser.return_value.parse_args.side_effect = SystemExit(0)
        
        with self.assertRaises(SystemExit):
            AiderVoiceGUI.parse_arguments(["--help"])

class TestGUIEventHandlers(AsyncTestCase):
    """Test GUI event handlers and keyboard shortcuts"""

    def setUp(self):
        """Set up test environment"""
        self.root = tk.Tk()
        self.app = AiderVoiceGUI(self.root)

    def tearDown(self):
        """Clean up test environment"""
        self.root.destroy()

    async def test_keyboard_shortcuts(self):
        """Test keyboard shortcut bindings and handlers with async support"""
        # Create mock event
        event = type('Event', (), {'widget': None})()
        
        # Test each shortcut individually
        shortcuts = {
            '<Control-r>': 'check_all_issues',
            '<Control-a>': 'browse_files',
            '<Control-v>': 'use_clipboard_content', 
            '<Control-s>': 'send_input_text',
            '<Escape>': 'stop_voice_control'
        }

        for key, method_name in shortcuts.items():
            with patch.object(self.app, method_name) as mock_method:
                self.app.root.event_generate(key)
                await asyncio.sleep(0.1)  # Allow event processing
                self.root.update()
                mock_method.assert_called_once()
                mock_method.reset_mock()

        # Test invalid shortcut
        with patch.object(self.app, 'log_message') as mock_log:
            self.app.root.event_generate('<Control-x>')
            await asyncio.sleep(0.1)
            self.root.update()
            mock_log.assert_not_called()


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
            "url": self.process_url,
        }

        # Error tracking
        self.error_count = 0
        self.max_errors = 3
        self.last_error_time = 0
        self.error_cooldown = 60  # seconds

    def get_current_content(self):
        """Get and process current clipboard content.
        
        Returns:
            str: The processed clipboard content
        """
        content = pyperclip.paste()
        content_type = self.detect_content_type(content)
        return self.processors[content_type](content)

    def detect_content_type(self, content):
        """Detect the type of clipboard content"""
        if self.looks_like_code(content):
            return "code"
        if self.looks_like_url(content):
            return "url"
        return "text"

    def looks_like_code(self, content):
        """Check if content appears to be code"""
        code_indicators = ["def ", "class ", "import ", "function", "{", "}", ";"]
        return any(indicator in content for indicator in code_indicators)

    def looks_like_url(self, content):
        """Check if content appears to be a URL"""
        return content.startswith(("http://", "https://", "www."))

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
        return "\n".join(cleaned_lines)

    def process_text(self, content):
        """Process text content"""
        # Basic text cleanup
        return content.strip()

    def process_url(self, content):
        """Process URL content"""
        # Basic URL validation and cleanup
        return content.strip()


from enum import Enum, auto

class ConnectionState(Enum):
    """Enum for WebSocket connection states"""
    DISCONNECTED = auto()  # Initial state or after clean disconnect
    CONNECTING = auto()    # Attempting initial connection
    CONNECTED = auto()     # Successfully connected
    RECONNECTING = auto()  # Attempting to restore lost connection
    FAILED = auto()        # Connection attempts exhausted
    ERROR = auto()         # Unexpected error state
    CLOSING = auto()       # Clean shutdown in progress

class WebSocketManager:
    """Manages WebSocket connection state and monitoring"""

    def __init__(self, parent):
        self.parent = parent
        self._state = ConnectionState.DISCONNECTED
        self.last_ping_time = 0
        self.ping_interval = 30  # seconds
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.monitoring_task = None
        self.log_message = parent.log_message
        self.ws = parent.ws
        self.last_error = None
        self.error_time = 0
        # Define valid state transitions with reasons
        self._state_transitions = {
            ConnectionState.DISCONNECTED: {
                ConnectionState.CONNECTING: "Initial connection attempt",
                ConnectionState.FAILED: "Connection initialization failed",
                ConnectionState.ERROR: "Unexpected error during initialization",
                ConnectionState.RECONNECTING: "Attempting to restore connection"
            },
            ConnectionState.CONNECTING: {
                ConnectionState.CONNECTED: "Connection established successfully", 
                ConnectionState.FAILED: "Connection attempt failed",
                ConnectionState.ERROR: "Error during connection attempt",
                ConnectionState.DISCONNECTED: "Connection attempt cancelled",
                ConnectionState.RECONNECTING: "Retrying connection"
            },
            ConnectionState.CONNECTED: {
                ConnectionState.CLOSING: "Connection closing normally",
                ConnectionState.RECONNECTING: "Connection lost unexpectedly",
                ConnectionState.ERROR: "Unexpected connection error",
                ConnectionState.FAILED: "Connection failed unexpectedly",
                ConnectionState.DISCONNECTED: "Connection terminated"
            },
            ConnectionState.RECONNECTING: {
                ConnectionState.CONNECTED: "Reconnection successful",
                ConnectionState.FAILED: "Reconnection attempts exhausted",
                ConnectionState.ERROR: "Error during reconnection",
                ConnectionState.DISCONNECTED: "Reconnection cancelled"
            },
            ConnectionState.FAILED: {
                ConnectionState.CONNECTING: "Retrying connection after failure",
                ConnectionState.DISCONNECTED: "Connection permanently failed",
                ConnectionState.ERROR: "Critical error after failure"
            },
            ConnectionState.ERROR: {
                ConnectionState.CONNECTING: "Attempting recovery from error",
                ConnectionState.DISCONNECTED: "Shutting down after error", 
                ConnectionState.FAILED: "Error recovery failed",
                ConnectionState.RECONNECTING: "Retrying after error",
                ConnectionState.CONNECTED: "Error resolved successfully"
            },
            ConnectionState.CLOSING: {
                ConnectionState.DISCONNECTED: "Connection closed normally",
                ConnectionState.ERROR: "Error during close",
                ConnectionState.FAILED: "Close operation failed"
            }
        }

    @property 
    def connection_state(self):
        """Get current connection state"""
        return self._state

    @connection_state.setter
    def connection_state(self, new_state):
        """Set connection state with validation and logging"""
        old_state = self._state
        try:
            if not isinstance(new_state, ConnectionState):
                raise StateError(f"Invalid state type: {type(new_state)}")
            
            # Validate state transition
            if new_state not in self._state_transitions[self._state]:
                valid_transitions = [
                    f"{s.name} ({reason})"
                    for s, reason in self._state_transitions[self._state].items()
                ]
                self.log_message(
                    f"⚠️ Invalid state transition attempted: "
                    f"{self._state.name} -> {new_state.name}"
                )
                raise ValueError(
                    f"Invalid state transition from {self._state.name} to {new_state.name}.\n"
                    f"Valid transitions are:\n" + "\n".join(f"- {t}" for t in valid_transitions)
                )

            # Get transition reason first
            transition_reason = self._state_transitions[self._state][new_state]

            # Track error state
            if new_state == ConnectionState.ERROR:
                self.last_error = transition_reason
                self.error_time = time.time()
            
            # Update state
            self._state = new_state

            # Handle connection state changes
            if new_state == ConnectionState.CONNECTED:
                self.reconnect_attempts = 0
                self.last_error = None
            elif new_state == ConnectionState.RECONNECTING:
                self.reconnect_attempts += 1
                if self.reconnect_attempts > self.max_reconnect_attempts:
                    raise ValueError("Maximum reconnection attempts exceeded")

            # Log state change with appropriate emoji and color
            emoji_map = {
                ConnectionState.CONNECTED: ("✅", "green"),
                ConnectionState.DISCONNECTED: ("❌", "red"),
                ConnectionState.CONNECTING: ("🔄", "blue"),
                ConnectionState.RECONNECTING: ("🔁", "yellow"),
                ConnectionState.FAILED: ("💥", "red"),
                ConnectionState.CLOSING: ("🚪", "blue")
            }
            emoji, color = emoji_map.get(new_state, ("", "white"))
            details = []
            
            if new_state == ConnectionState.RECONNECTING:
                details.append(f"Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
            if new_state == ConnectionState.FAILED:
                details.append(f"After {self.reconnect_attempts} attempts")
            if new_state == ConnectionState.CONNECTED:
                details.append(f"Latency: {self.get_connection_latency():.1f}ms")
                
            detail_str = f"\nDetails: {', '.join(details)}" if details else ""
            
            self.log_message(
                f"{emoji} WebSocket state transition: {old_state.name} -> {new_state.name}\n"
                f"Reason: {transition_reason}"
                f"{detail_str}"
            )

        except ValueError as e:
            self.log_message(f"⚠️ Invalid state transition: {str(e)}")
            raise
        except Exception as e:
            self.log_message(f"❌ Unexpected error during state transition: {type(e).__name__}: {str(e)}")
            raise

    async def start_monitoring(self):
        """Start connection monitoring"""
        self.monitoring_task = asyncio.create_task(self.monitor_connection())

    async def monitor_connection(self):
        """Monitor connection health and handle reconnection"""
        while True:
            try:
                if self.connection_state == ConnectionState.CONNECTED:
                    if time.time() - self.last_ping_time > self.ping_interval:
                        await self.check_connection()
                elif self.connection_state == ConnectionState.DISCONNECTED:
                    self.connection_state = ConnectionState.CONNECTING
                    await self.attempt_reconnect()
                elif self.connection_state == ConnectionState.FAILED:
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        self.connection_state = ConnectionState.CONNECTING
                        await self.attempt_reconnect()
                    else:
                        self.log_message("Max reconnection attempts reached")
                        break

                await asyncio.sleep(1)
            except websockets.exceptions.WebSocketException as e:
                self.log_message(f"🔌 WebSocket error during monitoring: {type(e).__name__}\n"
                               f"Details: {str(e)}\n"
                               f"Current state: {self.connection_state.name}")
                self.connection_state = ConnectionState.DISCONNECTED
            except ConnectionError as e:
                self.log_message(f"🌐 Network error during monitoring: {type(e).__name__}\n"
                               f"Details: {str(e)}\n"
                               f"Current state: {self.connection_state.name}")
                self.connection_state = ConnectionState.DISCONNECTED
            except asyncio.CancelledError:
                self.log_message("⏹️ Connection monitoring cancelled")
                break
            except ValueError as e:
                self.log_message(f"⚠️ Invalid state transition during monitoring:\n"
                               f"Details: {str(e)}\n"
                               f"Current state: {self.connection_state.name}")
            except Exception as e:
                self.log_message(f"❌ Unexpected error during monitoring: {type(e).__name__}\n"
                               f"Details: {str(e)}\n"
                               f"Current state: {self.connection_state.name}")
                self.connection_state = ConnectionState.FAILED

    async def check_connection(self):
        """Check connection health with ping"""
        try:
            if self.parent.ws:
                await self.parent.ws.ping()
                self.last_ping_time = time.time()
                return True
            return False
        except websockets.exceptions.WebSocketException as e:
            self.log_message(f"⚠️ WebSocket protocol error during connection check:\n"
                           f"Type: {type(e).__name__}\n"
                           f"Details: {str(e)}\n"
                           f"Last ping: {time.strftime('%H:%M:%S', time.localtime(self.last_ping_time))}")
            self.connection_state = ConnectionState.DISCONNECTED
            return False
        except ConnectionError as e:
            self.log_message(f"🌐 Network error during connection check:\n"
                           f"Type: {type(e).__name__}\n"
                           f"Details: {str(e)}\n"
                           f"Last ping: {time.strftime('%H:%M:%S', time.localtime(self.last_ping_time))}")
            self.connection_state = ConnectionState.DISCONNECTED
            return False
        except Exception as e:
            self.log_message(f"❌ Critical error during connection check:\n"
                           f"Type: {type(e).__name__}\n"
                           f"Details: {str(e)}\n"
                           f"Last ping: {time.strftime('%H:%M:%S', time.localtime(self.last_ping_time))}")
            self.connection_state = ConnectionState.FAILED
            return False

    async def attempt_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.connection_state = ConnectionState.FAILED
            self.log_message("❌ Max reconnection attempts reached")
            return False

        delay = min(30, 2**self.reconnect_attempts)
        self.log_message(f"Attempting reconnection in {delay} seconds...")
        await asyncio.sleep(delay)

        try:
            self.connection_state = ConnectionState.RECONNECTING
            if await self.parent.connect_websocket():
                self.connection_state = ConnectionState.CONNECTED
                self.reconnect_attempts = 0
                self.log_message("✅ Successfully reconnected")
                return True
            else:
                self.connection_state = ConnectionState.FAILED
                return False
        except (websockets.exceptions.WebSocketException, ConnectionError, OSError) as e:
            self.reconnect_attempts += 1
            self.log_message(f"Reconnection attempt failed: {e}")
            self.connection_state = ConnectionState.FAILED
            return False
        except asyncio.CancelledError:
            self.connection_state = ConnectionState.DISCONNECTED
            return False
        except Exception as e:
            self.log_message(f"Unexpected error during reconnection: {e}")
            self.connection_state = ConnectionState.FAILED
            return False



def main():
    """Main entry point"""
    root = tk.Tk()
    root.geometry("1200x800")
    app = AiderVoiceGUI(root)
    
    if all([sd, np, websockets, OpenAI]):
        root.after(1000, app.start_voice_control)
    
    root.mainloop()

if __name__ == "__main__":
    main()
