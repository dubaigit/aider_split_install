import os
import sys
import argparse
import subprocess
import tempfile
import select
import re
import time
from queue import Queue  # Add this import
import queue
import json
import base64
import asyncio
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog  # Added filedialog import

# Optional imports with better error handling
try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("Warning: sounddevice module not found. Voice functionality will be disabled.")

try:
    import numpy as np
except ImportError:
    np = None
    print("Warning: numpy module not found. Voice functionality will be disabled.")

try:
    import pyaudio
except ImportError:
    pyaudio = None
    print("Warning: pyaudio module not found. Voice functionality will be disabled.")

try:
    import websockets
except ImportError:
    websockets = None
    print("Warning: websockets module not found. Voice functionality will be disabled.")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("Warning: openai module not found. Voice functionality will be disabled.")

try:
    import pyperclip
except ImportError:
    pyperclip = None
    print("Warning: pyperclip module not found. Clipboard functionality will be disabled.")

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

# Audio settings (only if pyaudio is available)
# Audio constants
CHUNK_SIZE = 1024
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16 if pyaudio else None
REENGAGE_DELAY_MS = 500
OPENAI_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

class ModernDarkTheme:
    """Modern dark theme colors and styles"""
    # Colors
    BG_DARK = "#1a1a1a"
    BG_MEDIUM = "#2d2d2d"
    BG_LIGHT = "#383838"
    FG_DARK = "#666666"
    FG_LIGHT = "#ffffff"
    ACCENT = "#007acc"
    ACCENT_LIGHT = "#3399ff"
    ERROR = "#ff3333"
    SUCCESS = "#33cc33"
    
    # Fonts
    MAIN_FONT = ("Segoe UI", 10)
    MONO_FONT = ("Cascadia Code", 10)
    
    @classmethod
    def apply_theme(cls, root: tk.Tk) -> None:
        """Apply the modern dark theme to all widgets"""
        style = ttk.Style()
        style.configure(".",
            background=cls.BG_DARK,
            foreground=cls.FG_LIGHT,
            font=cls.MAIN_FONT
        )
        
        # Configure styles for different widget types
        style.configure("Dark.TFrame", background=cls.BG_DARK)
        style.configure("Dark.TLabel", 
            background=cls.BG_DARK,
            foreground=cls.FG_LIGHT
        )
        style.configure("Dark.TButton",
            background=cls.BG_MEDIUM,
            foreground=cls.FG_LIGHT,
            borderwidth=0
        )
        # ... existing code ...

class AiderVoiceGUI:
    def __init__(self, root):
        self.root = root
        self.theme = ModernDarkTheme()
        self.theme.apply_theme(root)
        
        # Initialize basic attributes
        self.recording = False
        self.auto_mode = False
        self.audio_queue = queue.Queue()
        self.ws = None
        self.running = True
        self.aider_process = None
        self.temp_files = []
        self.fixing_issues = False
        self.added_files = set()
        self.current_context = {
            "files": set(),
            "last_command": None,
            "last_response": None
        }
        
        # Initialize asyncio loop
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()
        
        # Initialize queues
        self.audio_queue = Queue()
        self.terminal_queue = Queue()
        
        # Initialize terminal state
        self.terminal_running = False
        self.shell = None
        
        # Start terminal
        self.start_terminal()
        
        # Create main container with modern styling
        self.main_frame = ttk.Frame(self.root, style="Dark.TFrame", padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize GUI elements
        self.files_frame = ttk.Frame(self.main_frame, style="Dark.TFrame")
        self.issues_frame = ttk.Frame(self.main_frame, style="Dark.TFrame")
        self.input_frame = ttk.Frame(self.main_frame, style="Dark.TFrame")
        self.tabs = ttk.Notebook(self.main_frame, style="Dark.TNotebook")
        
        # Initialize text widgets
        self.files_list = tk.Listbox(self.files_frame, bg=self.theme.BG_DARK, fg=self.theme.FG_LIGHT)
        self.issues_text = scrolledtext.ScrolledText(self.issues_frame, bg=self.theme.BG_DARK, fg=self.theme.FG_LIGHT)
        self.input_text = scrolledtext.ScrolledText(self.input_frame, bg=self.theme.BG_DARK, fg=self.theme.FG_LIGHT)
        self.transcription_text = scrolledtext.ScrolledText(self.main_frame, bg=self.theme.BG_DARK, fg=self.theme.FG_LIGHT)
        self.terminal_input = ttk.Entry(self.main_frame, style="Dark.TEntry")
        self.status_label = ttk.Label(self.main_frame, text="Ready", style="Dark.TLabel")
        
        # Create and style the notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame, style="Dark.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create modern styled tabs
        self.create_main_tab()
        self.create_files_tab()
        self.create_terminal_tab()
        self.create_output_tab()
        
        # Configure grid weights
        self.configure_grid_weights()
        
        # Initialize audio components
        # Initialize audio with error handling
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            self.log_message(f"Error initializing audio: {e}")
            self.p = None
        self.audio_buffer = bytearray()  # Changed from bytes to bytearray
        self.mic_queue = queue.Queue()
        self.mic_on_at = 0
        self.mic_active = None
        self._stop_event = threading.Event()
        
        # Add performance settings
        self.log_frequency = 50  # Only log every 50th audio chunk
        self.log_counter = 0
        self.chunk_buffer = []  # Buffer for accumulating audio chunks
        self.chunk_buffer_size = 5  # Number of chunks to accumulate before sending
        
        # Initialize audio processing thread
        self.audio_thread = None
        
        # Configure grid weights
        self.files_frame.columnconfigure(0, weight=1)  # Make file list expand horizontally
        self.files_frame.rowconfigure(0, weight=1)  # Make file list expand vertically
        
        # Initialize file tracking
        self.added_files = set()
        self.current_context = {
            "files": set(),
            "last_command": None,
            "last_response": None
        }
        
        # Rest of initialization...

    def create_main_tab(self):
        """Create the main control tab with modern styling"""
        main_frame = ttk.Frame(self.notebook, style="Dark.TFrame", padding="5")
        self.notebook.add(main_frame, text="Controls")
        
        # Control buttons with modern styling
        controls = ttk.LabelFrame(main_frame, text="Actions", style="Dark.TFrame", padding="10")
        controls.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.voice_button = ttk.Button(
            controls,
            text="🎤 Start Voice",
            command=self.toggle_voice_control,
            style="Dark.TButton"
        )
        self.voice_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        # Add other modern styled controls
        # ... existing code with style="Dark.TButton" added ...

    def create_files_tab(self):
        """Create the files tab with modern styling"""
        files_frame = ttk.Frame(self.notebook, style="Dark.TFrame", padding="5")
        self.notebook.add(files_frame, text="Files")
        
        # Add modern styled file list
        # ... existing code with modern styling ...

    def create_terminal_tab(self):
        """Create the terminal tab with modern styling"""
        terminal_frame = ttk.Frame(self.notebook, style="Dark.TFrame", padding="5")
        self.notebook.add(terminal_frame, text="Terminal")
        
        self.terminal = scrolledtext.ScrolledText(
            terminal_frame,
            height=20,
            bg=self.theme.BG_DARK,
            fg=self.theme.FG_LIGHT,
            insertbackground=self.theme.FG_LIGHT,
            font=self.theme.MONO_FONT
        )
        self.terminal.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Add modern styled terminal input
        # ... existing code with modern styling ...

    def create_output_tab(self):
        """Create the output tab with modern styling"""
        output_frame = ttk.Frame(self.notebook, style="Dark.TFrame", padding="5")
        self.notebook.add(output_frame, text="Output")
        
        # Add modern styled output text
        # ... existing code with modern styling ...

    def configure_grid_weights(self):
        """Configure grid weights for modern layout"""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.notebook.columnconfigure(0, weight=1)
        self.notebook.rowconfigure(0, weight=1)

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
            
        try:
            self.recording = True
            self.voice_button.configure(text="🔴 Stop Voice Control")
            self.status_label.configure(text="Listening...")
            
            # Start audio streams with error handling
            try:
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
            except OSError as e:
                self.log_message(f"Error opening audio streams: {e}")
                self.log_message("Voice control will run without audio.")
                self.mic_stream = None
                self.spkr_stream = None
            
            # Connect to OpenAI WebSocket
            asyncio.run_coroutine_threadsafe(self.connect_websocket(), self.loop)
            
        except Exception as e:
            self.log_message(f"Error starting voice control: {e}")
            self.stop_voice_control()

    def stop_voice_control(self):
        """Stop voice control"""
        self.recording = False
        self.voice_button.configure(text="🎤 Start Voice Control")
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

    def _mic_callback(self, in_data, frame_count, time_info, status):
        """Microphone callback that queues audio chunks."""
        if status:
            self.log_message(f"Audio input error: {status}")
            return (None, pyaudio.paContinue)
            
        if time.time() > self.mic_on_at:
            if not self.mic_active:
                self.log_message('🎙️🟢 Mic active')
                self.mic_active = True
            self.mic_queue.put(in_data)
        else:
            if self.mic_active:
                self.log_message('🎙️🔴 Mic suppressed')
                self.mic_active = False
                
        return (None, pyaudio.paContinue)

    def _process_audio_thread(self):
        """Process audio in a separate thread"""
        while self.recording:
            try:
                # Accumulate chunks
                chunks = []
                while len(chunks) < self.chunk_buffer_size:
                    try:
                        chunk = self.mic_queue.get_nowait()
                        chunks.append(chunk)
                    except queue.Empty:
                        break
                
                if chunks and self.ws:
                    # Combine chunks and send
                    combined_chunk = b''.join(chunks)
                    asyncio.run_coroutine_threadsafe(
                        self.ws.send(json.dumps({
                            'type': 'input_audio_buffer.append',
                            'audio': base64.b64encode(combined_chunk).decode('utf-8')
                        })),
                        self.loop
                    )
            except Exception as e:
                self.log_message(f"Error in audio processing thread: {e}")
            
            time.sleep(0.01)  # Short sleep to prevent tight loop

    def _spkr_callback(self, in_data, frame_count, time_info, status):
        """Speaker callback that plays audio."""
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

    async def connect_websocket(self):
        """Connect to OpenAI's realtime websocket API"""
        try:
            self.ws = await websockets.connect(
                OPENAI_WEBSOCKET_URL,
                extra_headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            
            # Create context-aware instructions
            files_context = "\n".join([f"- {os.path.basename(f)}" for f in self.added_files])
            instructions = f"""
            You are an AI assistant that helps control the aider code assistant through voice commands.
            You have full access to execute any bash command in the terminal.
            
            Currently added files:
            {files_context or "No files added yet"}
            
            You can:
            1. Execute any bash command by:
               - Understanding the user's intent
               - Running appropriate commands
               - Analyzing the output
               - Suggesting next steps
            
            2. Manage files:
               - Add/remove files
               - Check issues
               - Analyze code
               - Handle clipboard content
            
            3. Use the terminal for:
               - Navigation (cd, ls, pwd)
               - File operations (cp, mv, rm)
               - Package management (pip, npm)
               - Git operations
               - Any other bash command
            
            Always:
            1. Confirm what action you're taking
            2. Provide clear feedback
            3. Handle errors gracefully
            4. Suggest related commands
            5. Monitor command output
            
            Keep track of:
            - Current directory
            - Command history
            - File states
            - Process status
            """
            
            # Define available functions
            functions = [
                {
                    "name": "execute_terminal_command",
                    "description": "Execute a command in the terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute"
                            },
                            "working_dir": {
                                "type": "string",
                                "description": "Working directory for the command (optional)"
                            }
                        },
                        "required": ["command"]
                    }
                },
                {
                    "name": "manage_files",
                    "description": "Manage files in the application",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "remove", "clear", "list"],
                                "description": "Action to perform on files"
                            },
                            "filename": {
                                "type": "string",
                                "description": "File to act upon (for add/remove)"
                            }
                        },
                        "required": ["action"]
                    }
                },
                {
                    "name": "check_issues",
                    "description": "Run code quality checks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tools": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["ruff", "mypy", "all"]
                                },
                                "description": "Which tools to run"
                            }
                        },
                        "required": ["tools"]
                    }
                }
            ]
            
            # Update session with functions
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "model": "gpt-4o",
                    "voice": "alloy",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 200,
                        "silence_duration_ms": 300
                    },
                    "temperature": 0.8,
                    "max_response_output_tokens": 2048,
                    "instructions": instructions,
                    "tools": functions
                }
            }))
            
            self.log_message("Connected to OpenAI realtime API")
            
            # Initialize response state
            self.response_active = False
            self.last_transcript_id = None
            self.audio_buffer = bytearray()  # Changed from bytes to bytearray
            self.last_audio_time = time.time()
            
            # Start message handling
            asyncio.create_task(self.handle_websocket_messages())
            asyncio.create_task(self.process_audio_queue())
            
        except Exception as e:
            self.log_message(f"Failed to connect to OpenAI: {e}")
            self.stop_voice_control()

    async def process_audio_queue(self):
        """Process audio queue and send to OpenAI"""
        accumulated_audio = b''
        
        while self.recording:
            try:
                # Get audio data
                while not self.mic_queue.empty():
                    accumulated_audio += self.mic_queue.get()
                
                # Send if we have enough data
                if len(accumulated_audio) >= CHUNK_SIZE * 4:  # About 200ms of audio
                    if self.ws:
                        await self.ws.send(json.dumps({
                            'type': 'input_audio_buffer.append',
                            'audio': base64.b64encode(accumulated_audio).decode('utf-8')
                        }))
                        accumulated_audio = b''
                        
            except Exception as e:
                self.log_message(f"Error processing audio: {e}")
                
            await asyncio.sleep(0.01)

    async def handle_websocket_messages(self):
        """Handle incoming websocket messages"""
        while self.ws and self.recording:
            try:
                message = await self.ws.recv()
                event = json.loads(message)
                
                event_type = event.get("type")
                
                if event_type == "function_call":
                    # Handle function calls
                    function_name = event.get("function", {}).get("name")
                    arguments = event.get("function", {}).get("arguments", {})
                    
                    if function_name == "execute_terminal_command":
                        output = self.execute_terminal_command(arguments["command"])
                        # Send result back
                        await self.send_function_result(function_name, output or "Command executed successfully")
                        
                    elif function_name == "manage_files":
                        action = arguments["action"]
                        if action == "list":
                            files = "\n".join([f"- {os.path.basename(f)}" for f in self.added_files])
                            await self.send_function_result(function_name, files or "No files added")
                        elif action == "add" and "filename" in arguments:
                            self.add_files([arguments["filename"]])
                            await self.send_function_result(function_name, f"Added file: {arguments['filename']}")
                        elif action == "remove" and "filename" in arguments:
                            self.remove_file_by_name(arguments["filename"])
                            await self.send_function_result(function_name, f"Removed file: {arguments['filename']}")
                        elif action == "clear":
                            self.clear_files()
                            await self.send_function_result(function_name, "Cleared all files")
                            
                    elif function_name == "check_issues":
                        tools = arguments["tools"]
                        if "all" in tools or "ruff" in tools:
                            await self.run_ruff_check()
                        if "all" in tools or "mypy" in tools:
                            await self.run_mypy_check()
                        await self.send_function_result(function_name, "Checks completed")
                
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
        """Process transcribed voice commands directly without confirmation"""
        # Extract command from text
        command = text.lower().strip()
        
        # Direct execution of common commands
        if command.startswith(('ls', 'dir', 'pwd', 'cd', 'pip', 'python', 'git')):
            output = self.execute_terminal_command(command)
            if output:
                self.log_message(output)
            return
            
        # File management commands
        if "list files" in command:
            output = self.execute_terminal_command("ls -la")
            if output:
                self.log_message(output)
            return
            
        elif "check issues" in command:
            await self.analyze_and_check_issues()
            return
            
        elif "clear files" in command:
            self.clear_files()
            return
            
        elif "remove file" in command:
            filename = command.split("remove file")[-1].strip()
            if filename:
                self.remove_file_by_name(filename)
            return
        
        # Execute any other command directly
        output = self.execute_terminal_command(command)
        if output:
            self.log_message(output)

    def remove_file_by_name(self, filename):
        """Remove a file by its name"""
        for file_path in list(self.added_files):
            if filename in os.path.basename(file_path).lower():
                self.added_files.remove(file_path)
                self.current_context["files"].remove(file_path)
                # Update listbox
                for i in range(self.files_list.size()):
                    if self.files_list.get(i).lower() == os.path.basename(file_path).lower():
                        self.files_list.delete(i)
                        break
                self.log_message(f"Removed file: {file_path}")
                asyncio.run_coroutine_threadsafe(
                    self.send_audio_response(f"Removed {os.path.basename(file_path)}"),
                    self.loop
                )
                return
        asyncio.run_coroutine_threadsafe(
            self.send_audio_response(f"Could not find file matching '{filename}'"),
            self.loop
        )

    async def analyze_current_files(self):
        """Analyze currently added files"""
        if not self.added_files:
            await self.send_audio_response("No files are currently added. Please add some files first.")
            return
            
        files_content = {}
        for file_path in self.added_files:
            try:
                with open(file_path, 'r') as f:
                    files_content[os.path.basename(file_path)] = f.read()
            except Exception as e:
                self.log_message(f"Error reading file {file_path}: {e}")
                continue
        
        if not files_content:
            await self.send_audio_response("Could not read any of the added files.")
            return
            
        analysis_prompt = f"""
        Please analyze these files and provide a summary:
        
        Files to analyze:
        {json.dumps(files_content, indent=2)}
        
        Please provide:
        1. Overview of each file's purpose
        2. Key functions and their roles
        3. Any potential issues or improvements
        4. Dependencies between files
        """
        
        await self.send_ai_analysis(analysis_prompt)

    async def analyze_and_check_issues(self):
        """Run checks and analyze issues"""
        if not self.added_files:
            await self.send_audio_response("No files are currently added. Please add some files first.")
            return
            
        self.issues_text.delete('1.0', tk.END)
        self.issues_text.insert(tk.END, "Running checks...\n\n")
        
        issues = []
        
        # Run ruff on specific files
        try:
            ruff_result = subprocess.run(
                ["ruff", "check"] + list(self.added_files),
                capture_output=True,
                text=True
            )
            issues.append(("Ruff", ruff_result.stdout or "No issues found!"))
        except Exception as e:
            issues.append(("Ruff", f"Error: {e}"))
        
        # Run mypy on specific files
        try:
            mypy_result = subprocess.run(
                ["mypy"] + list(self.added_files),
                capture_output=True,
                text=True
            )
            issues.append(("Mypy", mypy_result.stdout or "No issues found!"))
        except Exception as e:
            issues.append(("Mypy", f"Error: {e}"))
        
        # Update issues display
        for tool, output in issues:
            self.issues_text.insert(tk.END, f"=== {tool} Issues ===\n{output}\n\n")
        
        # Have AI analyze the issues
        analysis_prompt = f"""
        Please analyze these issues and provide a detailed explanation:
        
        {json.dumps(dict(issues), indent=2)}
        
        Please provide:
        1. Summary of each type of issue
        2. Severity and impact of each issue
        3. Recommended fixes
        4. Priority order for addressing issues
        """
        
        await self.send_ai_analysis(analysis_prompt)

    async def analyze_clipboard_content(self, files_content):
        """Analyze clipboard content in context of current files"""
        try:
            clipboard_content = pyperclip.paste()
            if not clipboard_content.strip():
                await self.send_audio_response("Clipboard is empty. Please copy some content first.")
                return
            
            analysis_prompt = f"""
            Please analyze this clipboard content in the context of the current files:
            
            Clipboard content:
            {clipboard_content}
            
            Current files context:
            {json.dumps(files_content, indent=2)}
            
            Please provide:
            1. Analysis of the clipboard content
            2. How it relates to current files
            3. Suggested actions or changes
            4. Potential impacts of changes
            """
            
            await self.send_ai_analysis(analysis_prompt)
            
        except Exception as e:
            self.log_message(f"Error analyzing clipboard content: {e}")

    def run_aider_with_clipboard(self):
        """Run aider using clipboard content"""
        if self.aider_process and self.aider_process.poll() is None:
            self.log_message("Aider is already running. Please wait for it to finish.")
            return

        try:
            clipboard_content = pyperclip.paste()
            if not clipboard_content.strip():
                self.log_message("Clipboard is empty. Please copy some content first.")
                return

            self.aider_process = subprocess.Popen(
                ["python", "aider_wrapper.py", "-c"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.log_message("Started aider with clipboard content")
            
            # Monitor process output
            self.root.after(100, self.check_aider_process)
            
        except Exception as e:
            self.log_message(f"Error running aider with clipboard: {e}")

    async def add_files_to_aider(self):
        """Add files from current directory to aider"""
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
                self.log_message(f"Added {len(files)} files to aider")
                self.check_aider_process()
            else:
                self.log_message("No code files found in current directory")
        except Exception as e:
            self.log_message(f"Error adding files to aider: {e}")

    def run_aider_with_files(self, files):
        """Run aider with selected files"""
        if self.aider_process and self.aider_process.poll() is None:
            self.log_message("Aider is already running. Please wait for it to finish.")
            return

        try:
            # Convert to list of full paths
            file_paths = [f for f in files]
            
            self.aider_process = subprocess.Popen(
                ["python", "aider_wrapper.py"] + file_paths,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,  # Use text mode for easier output handling
                bufsize=1  # Line buffered
            )
            
            self.log_message(f"Started aider with files: {', '.join(os.path.basename(f) for f in file_paths)}")
            
            # Start monitoring in a separate thread
            threading.Thread(
                target=self.monitor_aider_output,
                args=(self.aider_process,),
                daemon=True
            ).start()
            
        except Exception as e:
            self.log_message(f"Error running aider with files: {e}")

    def monitor_aider_output(self, process):
        """Monitor aider output in a separate thread"""
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    self.log_message(line.strip())
            
            # Process finished
            if process.returncode == 0:
                self.log_message("Aider completed successfully")
            else:
                self.log_message(f"Aider exited with error code: {process.returncode}")
                
                # Get any error output
                errors = process.stderr.read()
                if errors:
                    self.log_message(f"Errors:\n{errors}")
                    
        except Exception as e:
            self.log_message(f"Error monitoring aider output: {e}")

    def check_aider_process(self):
        """Check aider process status and output"""
        if not self.aider_process:
            return

        # Check for output
        while True:
            try:
                output = self.aider_process.stdout.readline()
                if not output:
                    break
                self.log_message(output.decode().strip())
            except Exception as e:
                self.log_message(f"Error reading aider output: {e}")
                break

        # Check if process has finished
        if self.aider_process.poll() is not None:
            if self.aider_process.returncode == 0:
                self.log_message("Aider completed successfully")
            else:
                self.log_message("Aider encountered an error")
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
        """Check for code issues and send to aider"""
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
                text=True
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
                text=True
            )
            if mypy_result.stdout:
                issues_found = True
                combined_issues.append("Mypy issues:\n" + mypy_result.stdout)
                self.log_message("Found mypy issues")
            
            if issues_found:
                self.log_message("Issues found, sending to aider...")
                issues_file = tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='aider_issues_', suffix='.txt')
                self.temp_files.append(issues_file.name)
                
                with open(issues_file.name, 'w') as f:
                    f.write("\n\n".join(combined_issues))
                
                self.aider_process = subprocess.Popen(
                    ["python", "aider_wrapper.py", "-i", issues_file.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                self.log_message("Started aider to fix issues")
                
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
        """Send text to OpenAI for voice response"""
        if self.ws:
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
            except Exception as e:
                self.log_message(f"Error sending audio response: {e}")

    def log_message(self, message):
        """Log a message to the GUI"""
        try:
            self.root.after(0, self._update_log, message)
        except Exception:
            print(message)  # Fallback to console if GUI update fails

    def _update_log(self, message):
        """Update log in GUI thread"""
        try:
            # Use transcription_text instead of output_text
            self.transcription_text.insert(tk.END, message + "\n")
            self.transcription_text.see(tk.END)
        except Exception as e:
            print(f"Error updating log: {e}")
            print(message)

    def browse_files(self):
        """Open file browser to add files"""
        files = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=(
                ("Python files", "*.py"),
                ("JavaScript files", "*.js"),
                ("All files", "*.*")
            )
        )
        if files:
            self.add_files(files)

    def add_files(self, files):
        """Add files to the list without starting aider"""
        for file in files:
            if file not in self.added_files:
                self.added_files.add(file)
                self.files_list.insert(tk.END, os.path.basename(file))
                self.log_message(f"Added file: {file}")
                
                # Update AI context
                self.current_context["files"].add(file)
                # Run update_ai_context through asyncio
                asyncio.run_coroutine_threadsafe(
                    self.update_ai_context(),
                    self.loop
                )

    def remove_selected_file(self):
        """Remove selected file from the list"""
        selection = self.files_list.curselection()
        if selection:
            index = selection[0]
            filename = self.files_list.get(index)
            self.files_list.delete(index)
            
            # Remove from both sets
            full_path = next(f for f in self.added_files if os.path.basename(f) == filename)
            self.added_files.remove(full_path)
            self.current_context["files"].remove(full_path)
            
            self.log_message(f"Removed file: {filename}")
            # Run update_ai_context through asyncio
            asyncio.run_coroutine_threadsafe(
                self.update_ai_context(),
                self.loop
            )

    def clear_files(self):
        """Clear all files from the list"""
        self.files_list.delete(0, tk.END)
        self.added_files.clear()
        self.current_context["files"].clear()
        self.log_message("Cleared all files")
        # Run update_ai_context through asyncio
        asyncio.run_coroutine_threadsafe(
            self.update_ai_context(),
            self.loop
        )

    async def update_ai_context(self):
        """Update AI's context with current files"""
        if self.ws:
            try:
                files_context = "\n".join([f"- {os.path.basename(f)}" for f in self.current_context["files"]])
                instructions = f"""
                You are an AI assistant that helps control the aider code assistant through voice commands.
                
                Currently added files:
                {files_context or "No files added yet"}
                
                Commands you understand:
                - Run aider with clipboard content
                - Add files to aider (from current directory)
                - Check for issues and send to aider
                - Summarize what happened when aider finishes
                - List current files
                - Remove file <filename>
                - Clear all files
                
                Always confirm what action you're taking and provide clear feedback.
                Keep track of the files that have been added and removed.
                When checking for issues, focus on the currently added files.
                """
                
                await self.ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "instructions": instructions
                    }
                }))
            except Exception as e:
                self.log_message(f"Error updating AI context: {e}")

    def check_all_issues(self):
        """Run both ruff and mypy checks"""
        self.issues_text.delete('1.0', tk.END)
        self.issues_text.insert(tk.END, "Running checks...\n\n")
        
        # Run ruff
        try:
            ruff_result = subprocess.run(
                ["ruff", "check", "."],
                capture_output=True,
                text=True
            )
            self.issues_text.insert(tk.END, "=== Ruff Issues ===\n")
            self.issues_text.insert(tk.END, ruff_result.stdout or "No issues found!\n")
            self.issues_text.insert(tk.END, "\n")
        except Exception as e:
            self.issues_text.insert(tk.END, f"Error running ruff: {e}\n")
        
        # Run mypy
        try:
            mypy_result = subprocess.run(
                ["mypy", "."],
                capture_output=True,
                text=True
            )
            self.issues_text.insert(tk.END, "=== Mypy Issues ===\n")
            self.issues_text.insert(tk.END, mypy_result.stdout or "No issues found!\n")
        except Exception as e:
            self.issues_text.insert(tk.END, f"Error running mypy: {e}\n")
        
        self.issues_text.see(tk.END)
        self.tabs.select(self.issues_frame)

    def use_clipboard_content(self):
        """Get content from clipboard and show in input text"""
        try:
            content = pyperclip.paste()
            self.input_text.delete('1.0', tk.END)
            self.input_text.insert('1.0', content)
            self.tabs.select(self.input_frame)
            self.log_message("Clipboard content loaded into input tab")
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

    def update_transcription(self, text, is_assistant=False):
        """Update transcription with new text"""
        prefix = "🤖 Assistant: " if is_assistant else "🎤 You: "
        timestamp = time.strftime("%H:%M:%S")
        self.transcription_text.insert(tk.END, f"\n[{timestamp}] {prefix}{text}\n")
        self.transcription_text.see(tk.END)
        
        # Also update the log
        self.log_message(f"{prefix}{text}")

    def cleanup_terminal(self):
        """Clean up terminal resources"""
        try:
            self.terminal_running = False
            if hasattr(self, 'shell') and self.shell:
                self.shell.terminate()
                self.shell.wait(timeout=1)
            if hasattr(self, 'master_fd') and self.master_fd is not None:
                os.close(self.master_fd)
            if hasattr(self, 'slave_fd') and self.slave_fd is not None:
                os.close(self.slave_fd)
        except Exception as e:
            self.log_message(f"Error cleaning up terminal: {e}")

    def update_terminal_display(self):
        """Update terminal display from queue"""
        try:
            while not self.terminal_queue.empty():
                output = self.terminal_queue.get_nowait()
                if output:
                    self.terminal.insert(tk.END, output)
                    self.terminal.see(tk.END)
        except Exception as e:
            self.log_message(f"Error updating terminal display: {e}")

    def start_terminal(self):
        """Start terminal process"""
        try:
            # Initialize terminal state
            self.terminal_running = True
            
            # Start shell process with proper configuration
            self.shell = subprocess.Popen(
                ['/bin/bash'] if os.name == 'posix' else ['cmd.exe'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                env=dict(os.environ, TERM='xterm-color'),
                cwd=os.getcwd()
            )
            
            # Initial terminal prompt
            self.terminal.insert(tk.END, f"Terminal started in: {os.getcwd()}\n$ ")
            
            # Set up shell environment
            if os.name == 'posix':
                self.execute_terminal_command("export PS1='$ '")  # Simple prompt
                self.execute_terminal_command("export TERM=xterm-color")  # Enable color
            
            self.log_message("Terminal started successfully")
            
        except Exception as e:
            self.log_message(f"Error starting terminal: {e}")

    def execute_terminal_command(self, command):
        """Execute command in terminal and return output"""
        try:
            if not self.shell:
                self.log_message("Terminal not available")
                return None
            
            # Write command to terminal
            self.shell.stdin.write(f"{command}\n")
            self.shell.stdin.flush()
            
            # Show command in terminal
            self.terminal.insert(tk.END, f"$ {command}\n")
            
            # Read output immediately
            output = []
            
            # Read stdout with timeout
            start_time = time.time()
            while time.time() - start_time < 2:  # 2 second timeout
                line = self.shell.stdout.readline()
                if not line:
                    break
                output.append(line.strip())
                self.terminal.insert(tk.END, line)
                self.terminal.see(tk.END)
            
            # Read stderr
            while True:
                line = self.shell.stderr.readline()
                if not line:
                    break
                output.append(f"Error: {line.strip()}")
                self.terminal.insert(tk.END, f"Error: {line}")
                self.terminal.see(tk.END)
            
            # Return combined output
            return "\n".join(output) if output else None
            
        except Exception as e:
            error_msg = f"Error executing command: {e}"
            self.log_message(error_msg)
            self.terminal.insert(tk.END, f"{error_msg}\n")
            self.terminal.see(tk.END)
            return error_msg

    def add_to_files_from_terminal(self, filepath):
        """Add file to aider from terminal path"""
        if os.path.exists(filepath):
            self.add_files([filepath])
        else:
            self.terminal.insert(tk.END, f"Error: File not found: {filepath}\n")

    def get_terminal_output(self):
        """Get accumulated terminal output"""
        output = self.terminal.get('1.0', tk.END)
        return output

    def run_with_aider(self):
        """Run aider with currently added files"""
        if not self.added_files:
            self.log_message("No files added. Please add files first.")
            return

        if self.aider_process and self.aider_process.poll() is None:
            self.log_message("Aider is already running. Please wait for it to finish.")
            return

        try:
            # Convert to list of full paths
            file_paths = list(self.added_files)
            
            self.aider_process = subprocess.Popen(
                ["python", "aider_wrapper.py"] + file_paths,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.log_message(f"Started aider with files: {', '.join(os.path.basename(f) for f in file_paths)}")
            
            # Start monitoring in a separate thread
            threading.Thread(
                target=self.monitor_aider_output,
                args=(self.aider_process,),
                daemon=True
            ).start()
            
        except Exception as e:
            self.log_message(f"Error running aider with files: {e}")

    def execute_command(self, event=None):
        """Execute command from terminal input"""
        command = self.terminal_input.get().strip()
        if not command:
            return
        
        # Clear input
        self.terminal_input.delete(0, tk.END)
        
        # Execute command and get output
        output = self.execute_terminal_command(command)
        
        # If command was cd, update prompt
        if command.startswith('cd '):
            self.terminal.insert(tk.END, f"Current directory: {os.getcwd()}\n$ ")
        elif output:
            # Add prompt after output
            self.terminal.insert(tk.END, "$ ")
        else:
            # If no output, just add prompt
            self.terminal.insert(tk.END, "$ ")
        
        # Scroll to end
        self.terminal.see(tk.END)

    async def send_function_result(self, function_name, result):
        """Send function result back to OpenAI"""
        try:
            await self.ws.send(json.dumps({
                "type": "function_call.result",
                "function": {
                    "name": function_name,
                    "output": result
                }
            }))
        except Exception as e:
            self.log_message(f"Error sending function result: {e}")

    def initialize_audio(self):
        """Initialize audio components"""
        try:
            if pyaudio:
                self.p = pyaudio.PyAudio()
                self.audio_buffer = bytearray()
                self.mic_queue = queue.Queue()
                self.mic_on_at = 0
                self.mic_active = None
                self._stop_event = threading.Event()
            else:
                self.p = None
                self.log_message("Audio functionality not available - missing pyaudio")
        except Exception as e:
            self.log_message(f"Error initializing audio: {e}")
            self.p = None

    async def run_ruff_check(self):
        """Run ruff code checks"""
        try:
            result = subprocess.run(
                ["ruff", "check", "."],
                capture_output=True,
                text=True
            )
            return result.stdout or "No issues found!"
        except Exception as e:
            return f"Error running ruff: {e}"

    async def run_mypy_check(self):
        """Run mypy type checks"""
        try:
            result = subprocess.run(
                ["mypy", "."],
                capture_output=True,
                text=True
            )
            return result.stdout or "No issues found!"
        except Exception as e:
            return f"Error running mypy: {e}"

    async def send_ai_analysis(self, prompt):
        """Send analysis request to AI and handle response"""
        if self.ws:
            try:
                await self.ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": prompt
                        }]
                    }
                }))
                
                # Create response to generate analysis
                await self.ws.send(json.dumps({
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"]
                    }
                }))
                
            except Exception as e:
                self.log_message(f"Error sending analysis request: {e}")

def get_clipboard_content():
    """Get content from clipboard"""
    try:
        return pyperclip.paste()
    except Exception as e:
        print(f"Error getting clipboard content: {e}")
        return ""

def read_file_content(filename):
    """Read content from a file"""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""

def create_message_content(instructions, file_contents):
    """Create message content from instructions and files"""
    content = []
    if instructions:
        content.append("Instructions:")
        content.append(instructions)
        content.append("\n")
    
    if file_contents:
        content.append("File contents:")
        for filename, text in file_contents.items():
            content.append(f"\n{filename}:")
            content.append(text)
    
    return "\n".join(content)

def handle_aider_prompts(process):
    """Handle interactive prompts from aider"""
    try:
        while True:
            # Check for output
            output = process.stdout.readline()
            if not output:
                break
            print(output, end='')
            
            # Check for prompts that need response
            if "Do you want to proceed?" in output:
                process.stdin.write("y\n")
                process.stdin.flush()
    except Exception as e:
        print(f"Error handling aider prompts: {e}")

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
        try:
            # Initialize root window
            root = tk.Tk()
            root.title("Aider Voice Assistant")
            
            # Set initial window size
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.8)
            root.geometry(f"{window_width}x{window_height}")
            
            try:
                # Create app instance
                app = AiderVoiceGUI(root)
                if args.auto:
                    app.auto_mode = True
                    print("Auto mode enabled")
                
                # Start mainloop
                root.mainloop()
                
            except Exception as e:
                print(f"Error initializing application: {e}")
                root.destroy()
                return
                
        except Exception as e:
            print(f"Error creating window: {e}")
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
    print("Dark mode: Enabled")
    print(f"Chat mode: {args.chat_mode}")
    print(f"Suggest shell commands: {args.suggest_shell_commands}")

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

    # Get instructions content
    instructions = ""
    if args.clipboard:
        instructions = get_clipboard_content()
    elif args.instructions:
        instructions = read_file_content(args.instructions)

    # Read file contents
    file_contents = {filename: read_file_content(filename) for filename in args.filenames}

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
        "--chat-mode", args.chat_mode,
        "--message-file", temp_message_file.name,
    ]

    if args.suggest_shell_commands:
        aider_command.append("--suggest-shell-commands")

    if args.model:
        aider_command.extend(["--model", args.model])

    aider_command.extend(args.filenames)

    print("\nExecuting aider command:")
    print(" ".join(aider_command))

    try:
        process = subprocess.Popen(
            aider_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )

        handle_aider_prompts(process)

        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, aider_command)
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

