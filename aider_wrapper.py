import os
import sys
import argparse
import subprocess
import tempfile
import select
import re
import shutil
import time
import queue
import json
import base64
import asyncio
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import pyaudio

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

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm module not found. Progress bar functionality will be disabled.")
    tqdm = None

try:
    import git
except ImportError:
    print("Warning: git module not found. Git functionality will be disabled.")
    git = None

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
            raise AudioProcessingError(f"Error combining chunks: {e}")
            
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

class AiderVoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aider Voice Assistant")
        
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
        self.audio_queue = queue.Queue()
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
        self.main_frame = ttk.Frame(root, padding="10")
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
        self.audio_queue = queue.Queue()
        self.ws = None
        self.running = True
        self.client = OpenAI()
        self.aider_process = None
        self.temp_files = []
        self.fixing_issues = False
        
        # Initialize asyncio loop
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()
        
        # Initialize audio components
        self.p = pyaudio.PyAudio()
        self.audio_buffer = bytearray()  # Changed from bytes to bytearray
        self.mic_queue = queue.Queue()
        self.mic_on_at = 0
        self.mic_active = False
        self._stop_event = threading.Event()
        
        # Add performance settings
        self.log_frequency = 50  # Only log every 50th audio chunk
        self.log_counter = 0
        self.chunk_buffer = []  # Buffer for accumulating audio chunks
        self.chunk_buffer_size = 5  # Number of chunks to accumulate before sending
        
        # Initialize audio processing thread
        self.audio_thread = None
        
        # Automatically start voice control
        self.start_voice_control()
    
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
    
    async def _mic_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> tuple[None, int]:
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
    
    async def _spkr_callback(self, in_data: bytes, frame_count: int, time_info: dict, status: int) -> tuple[bytes, int]:
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
            if not self.mic_queue.empty():
                mic_chunk = self.mic_queue.get()
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
            except Exception as e:
                self.log_message(f"Error handling websocket message: {e}")
                self.log_message(f"Event content: {json.dumps(event, indent=2)}")
    
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
    
    def run_aider_with_clipboard(self):
        """Run Aider using clipboard content"""
        if self.aider_process and self.aider_process.poll() is None:
            self.log_message("Aider is already running. Please wait for it to finish.")
            return
            
        try:
            clipboard_content = pyperclip.paste()
            if not clipboard_content.strip():
                self.log_message("Clipboard is empty. Please copy some content first.")
                return
                
            # Analyze clipboard content
            analysis = self.analyze_clipboard_content(clipboard_content)
            self.interface_state['clipboard_history'].append({
                'content': clipboard_content,
                'analysis': analysis,
                'timestamp': time.time()
            })
            
            # Log analysis insights
            self.log_message("Clipboard Content Analysis:")
            self.log_message(f"- Type: {analysis['type']}")
            self.log_message(f"- Length: {analysis['length']} characters")
            self.log_message(f"- Purpose: {analysis['purpose']}")
            
            if analysis['recommendations']:
                self.log_message("Recommendations:")
                for rec in analysis['recommendations']:
                    self.log_message(f"- {rec}")

            # Start Aider process with clipboard content
            self.aider_process = subprocess.Popen(
                ["python", "aider_wrapper.py", "-c"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.log_message("Started Aider with clipboard content")
            
            # Monitor process output
            self.root.after(100, self.check_aider_process)
            
        except Exception as e:
            self.log_message(f"Error running Aider with clipboard: {e}")
    
    async def add_files_to_aider(self):
        """Add files from current directory to Aider"""
        if self.aider_process and self.aider_process.poll() is None:
            self.log_message("Aider is already running. Please wait for it to finish.")
            return

        try:
            files = [f for f in os.listdir('.') if f.endswith(('.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx'))]
            if files:
                self.aider_process = subprocess.Popen(
                    ["python", "aider_wrapper.py"] + files,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE
                )
                self.log_message(f"Added {len(files)} files to Aider")
                self.check_aider_process()
            else:
                self.log_message("No code files found in current directory")
        except Exception as e:
            self.log_message(f"Error adding files to Aider: {e}")
    
    def run_aider_with_files(self, files):
        """Run Aider with selected files"""
        if self.aider_process and self.aider_process.poll() is None:
            self.log_message("Aider is already running. Please wait for it to finish.")
            return

        try:
            self.aider_process = subprocess.Popen(
                ["python", "aider_wrapper.py"] + list(files),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            self.log_message(f"Started Aider with files: {', '.join(files)}")
            self.check_aider_process()
        except Exception as e:
            self.log_message(f"Error running Aider with files: {e}")
    
    def analyze_current_state(self):
        """Analyze the current state of the interface and files"""
        analysis = {
            'files': {},
            'issues': [],
            'interface_state': {},
            'recommendations': []
        }
        
        # Analyze loaded files
        for filename, content in self.interface_state['files'].items():
            analysis['files'][filename] = {
                'size': len(content),
                'type': self.detect_file_type(filename),
                'issues': self.analyze_file_issues(filename, content)
            }
        
        # Analyze interface state
        analysis['interface_state'] = {
            'has_unsaved_changes': bool(self.aider_process),
            'active_tasks': bool(self.fixing_issues),
            'available_actions': self.get_available_actions()
        }
        
        # Generate recommendations
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        return analysis
    
    def detect_file_type(self, filename):
        """Detect file type and relevant analysis approach"""
        ext = filename.split('.')[-1].lower()
        return {
            'py': 'python',
            'js': 'javascript',
            'html': 'html',
            'css': 'css',
            'json': 'json'
        }.get(ext, 'unknown')
    
    def analyze_file_issues(self, filename, content):
        """Analyze a file for potential issues"""
        issues = []
        
        # Basic analysis
        if not content.strip():
            issues.append(f"{filename} is empty")
        
        # Language-specific analysis
        file_type = self.detect_file_type(filename)
        if file_type == 'python':
            issues.extend(self.analyze_python_file(content))
        elif file_type == 'javascript':
            issues.extend(self.analyze_javascript_file(content))
            
        return issues
    
    def get_available_actions(self):
        """Determine available actions based on current state"""
        actions = []
        
        if self.interface_state['files']:
            actions.append('analyze_files')
            actions.append('check_issues')
            
        if self.interface_state['issues']:
            actions.append('fix_issues')
            
        if self.interface_state['clipboard_history']:
            actions.append('use_clipboard')
            
        return actions
    
    def generate_recommendations(self, analysis):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # File-based recommendations
        for filename, file_analysis in analysis['files'].items():
            if file_analysis['issues']:
                recommendations.append(f"Fix issues in {filename}")
                
        # State-based recommendations
        if analysis['interface_state']['has_unsaved_changes']:
            recommendations.append("Save current changes")
            
        if not analysis['files']:
            recommendations.append("Add files to analyze")
            
        return recommendations

    def check_aider_process(self):
        """Check Aider process status and output"""
        if not self.aider_process:
            return

        # Check for output
        while True:
            try:
                output = self.aider_process.stdout.readline()
                if not output:
                    break
                decoded = output.decode().strip()
                self.log_message(decoded)
                self.interface_state['aider_output'].append(decoded)
            except Exception as e:
                self.log_message(f"Error reading Aider output: {e}")
                break

        # Check if process has finished
        if self.aider_process.poll() is not None:
            if self.aider_process.returncode == 0:
                self.summarize_aider_session()
            else:
                self.log_message("⚠️ Aider encountered an error")
                self.summarize_aider_errors()
            self.aider_process = None
        else:
            # Process still running, check again later
            self.root.after(100, self.check_aider_process)
    
    def auto_check_issues(self):
        """Automatically check for issues in auto mode"""
        if not self.auto_mode:
            return
        
        if self.aider_process and self.aider_process.poll() is None:
            # Aider is still running, check again later
            self.root.after(1000, self.auto_check_issues)
            return
        
        # Run checks
        asyncio.run_coroutine_threadsafe(self.check_for_issues(), self.loop)
    
    async def check_for_issues(self):
        """Check for code issues and send to Aider"""
        if self.fixing_issues:
            self.log_message("Still working on previous issues. Please wait.")
            return

        if self.aider_process and self.aider_process.poll() is None:
            self.log_message("Aider is already running. Please wait for it to finish.")
            return

        self.fixing_issues = True
        try:
            issues_found = False
            combined_issues = []
            
            # Run ruff check
            self.log_message("Running ruff check...")
            ruff_result = subprocess.run(
                ["ruff", "check", "."],
                capture_output=True,
                text=True,
                check=False  # Don't raise on ruff findings
            )
            if ruff_result.stdout:
                issues_found = True
                combined_issues.append("Ruff issues:\n" + ruff_result.stdout)
                self.log_message("Found ruff issues")
            
            # Run mypy check with delay
            self.log_message("Running mypy check...")
            await asyncio.sleep(2)  # Wait for filesystem to settle
            mypy_result = subprocess.run(
                ["mypy", "."],
                capture_output=True,
                text=True,
                check=False  # Don't raise on type check findings
            )
            if mypy_result.stdout:
                issues_found = True
                combined_issues.append("Mypy issues:\n" + mypy_result.stdout)
                self.log_message("Found mypy issues")
            
            if issues_found:
                self.log_message("Issues found, sending to Aider...")
                issues_file = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='aider_issues_', suffix='.txt')
                self.temp_files.append(issues_file.name)
                
                with open(issues_file.name, 'w') as f:
                    f.write("\n\n".join(combined_issues))
                
                self.aider_process = subprocess.Popen(
                    ["python", "aider_wrapper.py", "-i", issues_file.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                self.log_message("Started Aider to fix issues")
                
                # Monitor process output
                self.root.after(100, self.check_aider_process)
            else:
                self.log_message("No issues found in the code")
                if self.auto_mode:
                    # Schedule next check
                    self.root.after(5000, self.auto_check_issues)
            
        except Exception as e:
            self.log_message(f"Error checking for issues: {e}")
        finally:
            self.fixing_issues = False
            # Cleanup temp files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
            self.temp_files = []
    
    async def send_audio_response(self, text):
        """Send text to OpenAI for voice response with improved error handling."""
        if not self.ws:
            self.log_message("⚠️ Cannot send audio response - WebSocket not connected")
            return
            
        try:
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": text
                    }]
                }
            }))
            self.log_message("✅ Voice response sent successfully")
        except websockets.exceptions.ConnectionClosed:
            self.log_message("❌ WebSocket connection closed - reconnecting...")
            await self.connect_websocket()
        except Exception as e:
            self.log_message(f"❌ Error sending audio response: {e}")
    
    def log_message(self, message):
        """Log a message to the GUI"""
        try:
            self.root.after(0, self._update_log, message)
        except Exception:
            print(message)  # Fallback to console if GUI update fails
    
    def _update_log(self, message):
        """Update log in GUI thread"""
        try:
            self.transcription_text.insert(tk.END, message + "\n")
            self.transcription_text.see(tk.END)
        except Exception as e:
            print(f"Error updating log: {e}")
            print(message)
    
    def browse_files(self):
        """Open file browser to add files to interface"""
        files = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=(
                ("Python files", "*.py"),
                ("JavaScript files", "*.js"),
                ("All files", "*.*")
            )
        )
        if files:
            added_files = []
            total_imports = 0
            total_classes = 0
            total_functions = 0
            
            for file in files:
                try:
                    content = self.read_file_content(file)
                    if content is not None:
                        self.interface_state['files'][file] = content
                        self.files_listbox.insert(tk.END, file)
                        added_files.append(file)
                        
                        # Analyze the file based on its type
                        file_ext = os.path.splitext(file)[1].lower()
                        analysis_result = ""
                        if file_ext == '.py':
                            analysis = self.analyze_python_file(content)
                            if analysis:
                                analysis_lines = analysis.split('\n')
                                for line in analysis_lines:
                                    if 'imports' in line:
                                        total_imports += int(line.split(':')[1])
                                    elif 'classes' in line:
                                        total_classes += int(line.split(':')[1])
                                    elif 'functions' in line:
                                        total_functions += int(line.split(':')[1])
                                analysis_result = analysis
                        elif file_ext in ['.js', '.jsx']:
                            analysis_result = self.analyze_javascript_file(content)
                            
                        self.log_message(f"Added and analyzed file: {file}")
                        if analysis_result:
                            self.log_message(f"Analysis results for {file}:")
                            self.log_message(analysis_result)
                            
                except Exception as e:
                    self.log_message(f"Error adding file {file}: {e}")
            
            if added_files:
                # Create a more detailed and conversational response
                if len(added_files) == 1:
                    filename = os.path.basename(added_files[0])
                    response = (
                        f"I see you've added {filename}. "
                        f"I found {total_imports} imports, {total_classes} classes, and {total_functions} functions. "
                        "Would you like me to analyze it for potential issues or help you make any specific changes?"
                    )
                else:
                    response = (
                        f"I see you've added {len(added_files)} files. "
                        f"In total, I found {total_imports} imports, {total_classes} classes, and {total_functions} functions. "
                        "Would you like me to check them for any issues or help you with specific modifications?"
                    )
                
                # Ensure the websocket is connected before sending
                if self.ws:
                    asyncio.run_coroutine_threadsafe(
                        self.send_audio_response(response),
                        self.loop
                    )
                    self.log_message("🗣️ Sending voice response: " + response)
                else:
                    self.log_message("⚠️ WebSocket not connected - voice response not sent")
    
    def analyze_python_file(self, content):
        """Analyze Python file content"""
        analysis = []
        
        # Check for imports
        imports = re.findall(r'^import\s+.*$|^from\s+.*\s+import\s+.*$', content, re.MULTILINE)
        if imports:
            analysis.append("Found imports: " + str(len(imports)))
            
        # Check for classes
        classes = re.findall(r'^class\s+\w+.*:$', content, re.MULTILINE)
        if classes:
            analysis.append("Found classes: " + str(len(classes)))
            
        # Check for functions
        functions = re.findall(r'^def\s+\w+\s*\(.*\):$', content, re.MULTILINE)
        if functions:
            analysis.append("Found functions: " + str(len(functions)))
            
        # Check for TODO comments
        todos = re.findall(r'#\s*TODO:', content, re.IGNORECASE)
        if todos:
            analysis.append("Found TODOs: " + str(len(todos)))
            
        return "\n".join(analysis)
    
    def analyze_javascript_file(self, content):
        """Analyze JavaScript file content"""
        analysis = []
        
        # Check for imports/requires
        imports = re.findall(r'^import\s+.*$|^const.*require\(.*\)$', content, re.MULTILINE)
        if imports:
            analysis.append("Found imports/requires: " + str(len(imports)))
            
        # Check for classes
        classes = re.findall(r'^class\s+\w+.*{$', content, re.MULTILINE)
        if classes:
            analysis.append("Found classes: " + str(len(classes)))
            
        # Check for functions
        functions = re.findall(r'^(function\s+\w+|\w+\s*=\s*function|\w+\s*:\s*function)', content, re.MULTILINE)
        if functions:
            analysis.append("Found functions: " + str(len(functions)))
            
        # Check for React components
        components = re.findall(r'^const\s+\w+\s*=\s*\(\s*\)\s*=>\s*{', content, re.MULTILINE)
        if components:
            analysis.append("Found React components: " + str(len(components)))
            
        return "\n".join(analysis)
    
    def generate_file_summary(self, files):
        """Generate a detailed summary of added files"""
        summary = f"I've added {len(files)} files to the workspace. "
        
        # Group files by type
        file_types = {}
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            
        # Add file type breakdown
        type_summary = []
        for ext, count in file_types.items():
            type_name = {
                '.py': 'Python',
                '.js': 'JavaScript',
                '.jsx': 'React',
                '.html': 'HTML',
                '.css': 'CSS'
            }.get(ext, ext[1:].upper())
            type_summary.append(f"{count} {type_name}")
            
        summary += "These include " + ", ".join(type_summary) + ". "
        
        # Add analysis prompt
        summary += "Would you like me to analyze these files for potential issues?"
        
        return summary

    def remove_selected_file(self):
        """Remove selected file from the list"""
        selected_indices = self.files_listbox.curselection()
        for index in reversed(selected_indices):
            file = self.files_listbox.get(index)
            self.files_listbox.delete(index)
            if file in self.temp_files:
                self.temp_files.remove(file)
            self.log_message(f"Removed file: {file}")
    
    def analyze_clipboard_content(self, content):
        """Analyze clipboard content for context and purpose"""
        analysis = {
            'type': 'unknown',
            'length': len(content),
            'purpose': 'unknown',
            'recommendations': []
        }
        
        # Detect content type
        if content.strip().startswith(('def ', 'class ')):
            analysis['type'] = 'python_code'
            analysis['purpose'] = 'code_addition_or_modification'
        elif '<' in content and '>' in content:
            analysis['type'] = 'markup'
            analysis['purpose'] = 'template_or_structure'
        elif 'error' in content.lower() or 'exception' in content.lower():
            analysis['type'] = 'error_message'
            analysis['purpose'] = 'error_resolution'
        elif len(content.splitlines()) == 1:
            analysis['type'] = 'single_line'
            analysis['purpose'] = 'quick_edit'
        
        # Generate recommendations
        if analysis['type'] == 'error_message':
            analysis['recommendations'].append("Send to Aider for error analysis")
        elif analysis['type'] == 'python_code':
            analysis['recommendations'].append("Run code quality checks before sending to Aider")
        
        return analysis

    def check_all_issues(self):
        """Run both ruff and mypy checks with improved voice feedback"""
        self.issues_text.delete('1.0', tk.END)
        self.issues_text.insert(tk.END, "Running checks...\n\n")
        
        all_issues = []
        has_issues = False
        
        # Run ruff
        try:
            ruff_result = subprocess.run(
                ["ruff", "check", "."],
                capture_output=True,
                text=True,
                check=False
            )
            self.issues_text.insert(tk.END, "=== Ruff Issues ===\n")
            if ruff_result.stdout:
                has_issues = True
                all_issues.extend(ruff_result.stdout.splitlines())
            self.issues_text.insert(tk.END, ruff_result.stdout or "No issues found!\n")
            self.issues_text.insert(tk.END, "\n")
        except Exception as e:
            self.issues_text.insert(tk.END, f"Error running ruff: {e}\n")
        
        # Run mypy
        try:
            mypy_result = subprocess.run(
                ["mypy", "."],
                capture_output=True,
                text=True,
                check=False
            )
            self.issues_text.insert(tk.END, "=== Mypy Issues ===\n")
            if mypy_result.stdout:
                has_issues = True
                all_issues.extend(mypy_result.stdout.splitlines())
            self.issues_text.insert(tk.END, mypy_result.stdout or "No issues found!\n")
        except Exception as e:
            self.issues_text.insert(tk.END, f"Error running mypy: {e}\n")
        
        self.issues_text.see(tk.END)
        
        # Provide more detailed voice feedback
        if has_issues:
            issue_count = len(all_issues)
            issue_types = set()
            for issue in all_issues:
                if "error" in issue.lower():
                    issue_types.add("errors")
                elif "warning" in issue.lower():
                    issue_types.add("warnings")
                
            issue_type_str = " and ".join(issue_types)
            response = f"I've found {issue_count} {issue_type_str} in the code. Would you like me to help fix these issues using Aider, or would you prefer to review them first?"
            
            asyncio.run_coroutine_threadsafe(
                self.send_audio_response(response),
                self.loop
            )
        else:
            asyncio.run_coroutine_threadsafe(
                self.send_audio_response("I've checked the code and everything looks good! No issues were found. Is there anything specific you'd like me to help you with?"),
                self.loop
            )
    
    def use_clipboard_content(self):
        """Get content from clipboard and show in input text"""
        try:
            content = pyperclip.paste()
            self.input_text.delete('1.0', tk.END)
            self.input_text.insert('1.0', content)
            self.log_message("Clipboard content loaded into input")
        except Exception as e:
            self.log_message(f"Error getting clipboard content: {e}")
    
    def send_input_text(self):
        """Send input text to assistant for processing"""
        content = self.input_text.get('1.0', tk.END).strip()
        if content:
            self.log_message("Processing input text...")
            # Send to assistant for processing
            asyncio.run_coroutine_threadsafe(
                self.process_voice_command(content),
                self.loop
            )
        else:
            self.log_message("No input text to process")
    
    def list_added_files(self):
        """List all added files in the transcription window"""
        files = self.temp_files
        if files:
            self.log_message("Added Files:")
            for file in files:
                self.log_message(f" - {file}")
        else:
            self.log_message("No files added.")
    
    def navigate_to_directory(self, directory):
        """Navigate to a specified directory"""
        try:
            os.chdir(directory)
            self.log_message(f"Navigated to directory: {directory}")
        except Exception as e:
            self.log_message(f"Error navigating to directory '{directory}': {e}")
    
    def update_transcription(self, text, is_assistant=False):
        """Update transcription with new text"""
        prefix = "🤖 Assistant: " if is_assistant else "🎤 You: "
        timestamp = time.strftime("%H:%M:%S")
        self.transcription_text.insert(tk.END, f"\n[{timestamp}] {prefix}{text}\n")
        self.transcription_text.see(tk.END)
        
        # Also update the log
        self.log_message(f"{prefix}{text}")

    def read_file_content(self, filename: str) -> str | None:
        """Read file content with robust error handling.
        
        Args:
            filename: Path to the file to read
            
        Returns:
            str: File contents if successful, None if error occurred
            
        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file access is denied
            IOError: If file read fails
        """
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                if not content:
                    self.log_message(f"Warning: File {filename} is empty")
                return content
        except FileNotFoundError as e:
            self.log_message(f"Error: File {filename} not found: {e}")
            raise
        except PermissionError as e:
            self.log_message(f"Error: Permission denied accessing {filename}: {e}")
            raise
        except IOError as e:
            self.log_message(f"Error reading file {filename}: {e}")
            raise
        except Exception as e:
            self.log_message(f"Unexpected error reading {filename}: {e}")
            return None

    def analyze_python_file(self, content):
        """Analyze Python file content"""
        analysis = []
        
        # Check for imports
        imports = re.findall(r'^import\s+.*$|^from\s+.*\s+import\s+.*$', content, re.MULTILINE)
        if imports:
            analysis.append("Found imports: " + str(len(imports)))
            
        # Check for classes
        classes = re.findall(r'^class\s+\w+.*:$', content, re.MULTILINE)
        if classes:
            analysis.append("Found classes: " + str(len(classes)))
            
        # Check for functions
        functions = re.findall(r'^def\s+\w+\s*\(.*\):$', content, re.MULTILINE)
        if functions:
            analysis.append("Found functions: " + str(len(functions)))
            
        # Check for TODO comments
        todos = re.findall(r'#\s*TODO:', content, re.IGNORECASE)
        if todos:
            analysis.append("Found TODOs: " + str(len(todos)))
            
        return "\n".join(analysis)

    def analyze_javascript_file(self, content):
        """Analyze JavaScript file content"""
        analysis = []
        
        # Check for imports/requires
        imports = re.findall(r'^import\s+.*$|^const.*require\(.*\)$', content, re.MULTILINE)
        if imports:
            analysis.append("Found imports/requires: " + str(len(imports)))
            
        # Check for classes
        classes = re.findall(r'^class\s+\w+.*{$', content, re.MULTILINE)
        if classes:
            analysis.append("Found classes: " + str(len(classes)))
            
        # Check for functions
        functions = re.findall(r'^(function\s+\w+|\w+\s*=\s*function|\w+\s*:\s*function)', content, re.MULTILINE)
        if functions:
            analysis.append("Found functions: " + str(len(functions)))
            
        # Check for React components
        components = re.findall(r'^const\s+\w+\s*=\s*\(\s*\)\s*=>\s*{', content, re.MULTILINE)
        if components:
            analysis.append("Found React components: " + str(len(components)))
            
        return "\n".join(analysis)

    def summarize_aider_session(self):
        """Summarize the completed Aider session results."""
        self.log_message("\n=== Aider Session Summary ===")
        
        # Count files processed
        files_processed = len(self.interface_state['files'])
        self.log_message(f"📁 Files processed: {files_processed}")
        
        # Analyze output for changes
        changes = []
        for line in self.interface_state['aider_output']:
            if "Changed" in line or "Modified" in line:
                changes.append(line)
        
        if changes:
            self.log_message("\n🔄 Changes made:")
            for change in changes:
                self.log_message(f"  • {change}")
        else:
            self.log_message("ℹ️ No files were modified")
            
        # Look for any remaining issues
        issues = [line for line in self.interface_state['aider_output'] if "error" in line.lower()]
        if issues:
            self.log_message("\n⚠️ Remaining issues to address:")
            for issue in issues[:5]:  # Show top 5 issues
                self.log_message(f"  • {issue}")
            
        self.log_message("\n✅ Aider session completed successfully")

    def summarize_aider_errors(self):
        """Summarize errors encountered during Aider session."""
        self.log_message("\n=== Aider Error Summary ===")
        
        errors = []
        for line in self.interface_state['aider_output']:
            if any(err in line.lower() for err in ['error', 'exception', 'failed']):
                errors.append(line)
            
        if errors:
            self.log_message("❌ Errors encountered:")
            for error in errors:
                self.log_message(f"  • {error}")
        
        self.log_message("\nPlease check the logs above for more details")

def create_message_content(instructions, file_contents):
    existing_code = "\n\n".join([f"File: {filename}\n```\n{content}\n```" for filename, content in file_contents.items()])
    prompt = '''
    You are Claude Dev, a highly skilled software development assistant with extensive knowledge in many programming languages, frameworks, design patterns, and best practices. You can seamlessly switch between multiple specialized roles to provide comprehensive assistance in various aspects of software development. When switching roles, always announce the change explicitly to ensure clarity.

## Capabilities

### General Software Development
- Read and analyze code in various programming languages.
- Analyze the problem and create a list of tasks.
- After creating the list, use <thinking></thinking> tags to think and see if you need to add or remove any tasks.
- Work on the tasks one by one, don't skip any.
- Write clean, efficient, and well-documented code.
- Debug complex issues and provide detailed explanations.
- Offer architectural insights and design patterns.
- Implement best coding practices and standards.
- After finishing the task, use <thinking></thinking> tags to confirm if the change will fix the problem; if not, repeat the process.

### Specialized Roles

#### Expert Debugger
- Analyze error messages and stack traces.
- Identify root causes of bugs and performance issues.
- Suggest efficient debugging strategies.
- Provide step-by-step troubleshooting instructions.
- Recommend tools and techniques for effective debugging.

#### Professional Coder
- Write optimized, scalable, and maintainable code.
- Implement complex algorithms and data structures.
- Refactor existing code for improved performance and readability.
- Integrate third-party libraries and APIs effectively.
- Develop unit tests and implement test-driven development practices.

#### Code Reviewer
- Conduct thorough code reviews for quality and consistency.
- Identify potential bugs, security vulnerabilities, and performance bottlenecks.
- Suggest improvements in code structure, naming conventions, and documentation.
- Ensure adherence to coding standards and best practices.
- Provide constructive feedback to improve overall code quality.

#### UX/UI Designer
- Create intuitive and visually appealing user interfaces.
- Design responsive layouts for various devices and screen sizes.
- Implement modern design principles and patterns.
- Suggest improvements for user experience and accessibility.
- Provide guidance on color schemes, typography, and overall aesthetics.

## Rules

1. Work in your current working directory.
2. Always provide complete file content in your responses, regardless of the extent of changes.
3. When creating new projects, organize files within a dedicated project directory unless specified otherwise.
4. Consider the project type (e.g., Python, JavaScript, web application) when determining appropriate structure and files.
5. Ensure changes are compatible with the existing codebase and follow project coding standards.
6. Use markdown freely in your responses, including language-specific code blocks.
7. Do not start responses with affirmations like "Certainly", "Okay", "Sure", etc. Be direct and to the point.
8. When switching roles, explicitly mention the role you're assuming for clarity, even repeating this when changing roles or returning to a previous role.

## Code Regeneration Approach

1. Ensure that you maintain the overall structure and logic of the code, unless the changes explicitly modify them.
2. Pay special attention to proper indentation and formatting throughout the entire file.
3. If a requested change seems inconsistent or problematic, use your expertise to implement it in the most logical way possible.
4. Wrap the regenerated code in a Python markdown code block.

## Objective

Accomplish given tasks iteratively, breaking them down into clear steps:

1. Analyze the user's task and set clear, achievable goals.
2. Prioritize goals in a logical order.
3. Work through goals sequentially, utilizing your multi-role expertise as needed.
4. Before taking action, analyze the task within <thinking></thinking> tags:
    - Determine which specialized role is most appropriate for the current step.
    - Consider the context and requirements of the task.
    - Plan your approach using the capabilities of the chosen role.
5. Execute the planned actions, explicitly mentioning when switching roles for clarity.
6. Once the task is completed, present the result to the user.
7. If feedback is provided, use it to make improvements and iterate on the solution.

## Example *SEARCH/REPLACE block*

To make this change, we need to modify `main.py` and create a new file `hello.py`:

1. Make a new `hello.py` file with `hello()` in it.
2. Remove `hello()` from `main.py` and replace it with an import.

Here are the *SEARCH/REPLACE* blocks:

```
hello.py
<<<<<<< SEARCH.
=======
def hello():
    "print a greeting"

    print("hello")
>>>>>>> REPLACE.
```

```
main.py
<<<<<<< SEARCH.
def hello():
    "print a greeting"

    print("hello")
=======
from hello import hello
>>>>>>> REPLACE.
```

## *SEARCH/REPLACE block* Rules

1. The *FULL* file path alone on a line, verbatim. No bold asterisks, no quotes around it, no escaping of characters, etc.
2. The start of the search block: <<<<<<< SEARCH.
3. A contiguous chunk of lines to search for in the existing source code.
4. The dividing line: =======.
5. The lines to replace into the source code.
6. The end of the replace block: >>>>>>> REPLACE.
7. The closing fence: >>>>>>> REPLACE.

Use the *FULL* file path, as shown to you by the user.

Every *SEARCH* section must *EXACTLY MATCH* the existing file content, character for character, including all comments, docstrings, etc.
If the file contains code or other data wrapped/escaped in json/xml/quotes or other containers, you need to propose edits to the literal contents of the file, including the container markup.

*SEARCH/REPLACE* blocks will replace *all* matching occurrences.
Include enough lines to make the SEARCH blocks uniquely match the lines to change.
Keep *SEARCH/REPLACE* blocks concise.
Break large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.
Include just the changing lines, and a few surrounding lines if needed for uniqueness.
Do not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.

Only create *SEARCH/REPLACE* blocks for files that the user has added to the chat!

To move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.

Pay attention to which filenames the user wants you to edit, especially if they are asking you to create a new file.

If you want to put code in a new file, use a *SEARCH/REPLACE* block with:
- A new file path, including dir name if needed.
- An empty `SEARCH` section.
- The new file's contents in the `REPLACE` section.

    ###Task problem_description
    <task_description>
    {task}
    </problem_description>

'''
    content = f"{prompt}\n\n<problem_description>\n{instructions}\n</problem_description>"
    return content

def enhance_user_experience():
    """
    Enhance user experience with a progress bar and better error handling.
    """
    if tqdm is None:
        return

    for i in tqdm(range(10), desc="Preparing environment"):
        time.sleep(0.1)

def read_file_content(filename: str) -> str:
    """Read file content with robust error handling.
    
    Args:
        filename: Path to the file to read
        
    Returns:
        str: File contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file access is denied
        IOError: If file read fails
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            if not content:
                print(f"Warning: File {filename} is empty")
            return content
    except FileNotFoundError as e:
        print(f"Error: File {filename} not found: {e}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: Permission denied accessing {filename}: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading {filename}: {e}")
        sys.exit(1)

def get_clipboard_content():
    """
    Get content from clipboard with enhanced error handling and validation
    """
    if pyperclip is None:
        print("Error: Clipboard functionality requires pyperclip module")
        print("Install it with: pip install pyperclip")
        sys.exit(1)
        
    try:
        content = pyperclip.paste()
        if not content:
            print("Error: Clipboard is empty")
            sys.exit(1)
            
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive newlines
        while '\n\n\n' in content:
            content = content.replace('\n\n\n', '\n\n')
            
        content = content.strip()
        print(f"Successfully read {len(content)} characters from clipboard")
        return content
        
    except pyperclip.PyperclipException as e:
        print(f"Clipboard error: {e}")
        print("Make sure you have access to the system clipboard")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error accessing clipboard: {e}")
        sys.exit(1)

def handle_aider_prompts(process):
    while True:
        ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
        for stream in ready:
            line = stream.readline()
            if not line:
                sys.exit(0)

            print(line, end='', flush=True)

            if re.search(r'Add .+ to the chat\? \(Y\)es/\(N\)o \[Yes\]:', line):
                process.stdin.write('Y\n')
                process.stdin.flush()
            elif 'Add URL to the chat? (Y)es/(N)o/(A)ll/(S)kip all [Yes]:' in line:
                process.stdin.write('S\n')
                process.stdin.flush()
            elif 'Add URL to the chat? (Y)es/(N)o [Yes]:' in line:
                process.stdin.write('N\n')
                process.stdin.flush()


def main():
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

    args = parser.parse_args()

    if args.gui:
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        if args.auto:
            app.auto_mode = True
            print("Auto mode enabled - will automatically send ruff issues to aider")
        root.mainloop()  # Added mainloop call
        return

    if args.clipboard and args.instructions:
        parser.error("Cannot use both clipboard and instruction file. Choose one option.")
    elif not args.clipboard and not args.instructions and not args.filenames:
        parser.error("Must specify either clipboard (-c) or instruction file (-i) or provide filenames.")
        sys.exit(1)

    print("Running aider wrapper with the following parameters:")
    print(f"Filenames: {', '.join(args.filenames)}")
    print(f"Using clipboard: {args.clipboard}")
    print(f"Instructions file: {args.instructions}")
    print(f"Model: {args.model}")
    print(f"Dark mode: Enabled")
    print(f"Chat mode: {args.chat_mode}")
    print(f"Suggest shell commands: {args.suggest_shell_commands}")

    enhance_user_experience()

    if git is not None:
        try:
            repo = git.Repo(search_parent_directories=True)
            repo.git.add(update=True)
            repo.index.commit("Auto-commit before running aider")
            print("Git commit created successfully.")
        except git.exc.InvalidGitRepositoryError:
            print("Warning: Not a git repository. Skipping git commit.")
        except Exception as e:
            print(f"Error creating git commit: {e}")
            
class VoiceCommandProcessor:
    """Processes and validates voice commands"""
    def __init__(self, parent):
        self.parent = parent
        
    def preprocess_command(self, command):
        """Clean and normalize voice command"""
        if not command:
            return ""
        return command.strip().lower()
        
    def validate_command(self, command):
        """Validate command format and content"""
        if not command:
            return False
        return True

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

    def handle_git_functionality(self):
        """Handle git functionality with proper error handling"""
        if git is not None:
            try:
                repo = git.Repo(search_parent_directories=True)
                repo.git.add(update=True)
                repo.index.commit("Auto-commit before running aider")
                print("Git commit created successfully.")
            except git.exc.InvalidGitRepositoryError:
                print("Warning: Not a git repository. Skipping git commit.")
            except Exception as e:
                print(f"Error creating git commit: {e}")
        else:
            print("Warning: Git functionality is disabled. Skipping git commit.")

        # Get instructions content from parsed arguments
        instructions = ""
        if self.args.clipboard:
            instructions = get_clipboard_content()
        elif self.args.instructions:
            instructions = read_file_content(self.args.instructions)

        # Read file contents
        file_contents = {filename: read_file_content(filename) for filename in self.args.filenames}

        # Create message content
        message_content = create_message_content(instructions, file_contents)

        # Write message content to a temporary file
        temp_message_file = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='aider_wrapper_', suffix='.txt')
        try:
            temp_message_file.write(message_content)
            temp_message_file.close()
            print(f"Temporary message file: {temp_message_file.name}")
        except IOError as e:
            print(f"Error writing to temporary file: {e}")
            sys.exit(1)

        aider_command = [
            "aider",
            "--no-pretty",
            "--dark-mode",
            "--yes",
            "--chat-mode", self.args.chat_mode,
            "--message-file", temp_message_file.name,
        ]

        if self.args.suggest_shell_commands:
            aider_command.append("--suggest-shell-commands")

        if self.args.model:
            aider_command.extend(["--model", self.args.model])

        aider_command.extend(self.args.filenames)

        print("\nExecuting aider command:")
        print(" ".join(aider_command))

        try:
            # Verify aider is installed and accessible
            try:
                subprocess.run(["aider", "--version"], capture_output=True, check=True)
            except subprocess.CalledProcessError:
                print("Error: Unable to run aider. Please ensure it's installed correctly")
                print("Install with: pip install aider-chat")
                sys.exit(1)
            except FileNotFoundError:
                print("Error: aider command not found. Please install aider-chat")
                print("Install with: pip install aider-chat")
                sys.exit(1)

            print("\nStarting aider process...")
            process = subprocess.Popen(
                aider_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )

            print("Handling aider interaction...")
            handle_aider_prompts(process)

            print("Waiting for aider to complete...")
            rc = process.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, aider_command)
            print("Aider completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error executing aider command: {e}", file=sys.stderr)
            print("The specified model may not be supported or there might be an issue with the aider configuration.")
            print("Please check your aider installation and ensure the model is correctly specified.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}", file=sys.stderr)
            print("Please report this issue to the developers.")
            sys.exit(1)
        finally:
            try:
                os.unlink(temp_message_file.name)
            except OSError:
                pass

if __name__ == "__main__":
    main()
    def __init__(self, root):
        self.root = root
        self.root.title("Aider Voice Assistant")
        
        # Parse command line arguments
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
        
        self.args = parser.parse_args()
        
    def __del__(self) -> None:
        """Cleanup resources when object is deleted."""
        try:
            # Stop and close audio streams
            if hasattr(self, 'mic_stream'):
                try:
                    self.mic_stream.stop_stream()
                    self.mic_stream.close()
                except Exception as e:
                    print(f"Error closing mic stream: {e}")
                    
            if hasattr(self, 'spkr_stream'):
                try:
                    self.spkr_stream.stop_stream()
                    self.spkr_stream.close()
                except Exception as e:
                    print(f"Error closing speaker stream: {e}")
            
            # Terminate PyAudio
            if hasattr(self, 'p'):
                try:
                    self.p.terminate()
                except Exception as e:
                    print(f"Error terminating PyAudio: {e}")
            
            # Close websocket
            if hasattr(self, 'ws') and self.ws:
                try:
                    asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
                except Exception as e:
                    print(f"Error closing websocket: {e}")
                    
            # Clean up temp files
            for temp_file in getattr(self, 'temp_files', []):
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f"Error removing temp file {temp_file}: {e}")
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")
    def add_files(self, file_paths):
        """Add files to the assistant for analysis."""
        added_files = []
        for file in file_paths:
            content = self.read_file_content(file)
            if content is not None:
                self.interface_state['files'][file] = content
                self.files_listbox.insert(tk.END, file)
                self.log_message(f"Added file: {file}")
                added_files.append(file)
        return {"status": "success", "added_files": added_files}

    async def check_issues(self):
        """Check the added files for issues."""
        await self.check_all_issues()
        return {"status": "success"}

    def list_files(self):
        """List all currently added files."""
        files = list(self.interface_state['files'].keys())
        return {"status": "success", "files": files}
