import os
import sys
import argparse
import subprocess
import tempfile
import select
import re
import shutil
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
import sounddevice as sd
import numpy as np
import websockets
import asyncio
import json
import queue
import threading
from openai import OpenAI
import pyperclip
import wave
import base64

git = None
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

try:
    import pyperclip
except ImportError:
    print("Warning: pyperclip module not found. Clipboard functionality will be disabled.")
    pyperclip = None

# Audio settings
SAMPLE_RATE = 24000  # Changed to 24kHz as per OpenAI docs
CHANNELS = 1
CHUNK_SIZE = 1024
OPENAI_WEBSOCKET_URL = "wss://api.openai.com/v1/realtime"

class AiderVoiceGUI:
    def __init__(self, tk_root):
        self.root = tk_root
        self.root.title("Aider Voice Assistant")
        self.root.geometry("800x600")
        self.setup_gui()
        
    def setup_gui(self):
def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()

def create_message_content(instructions, file_contents):
    existing_code = "\n\n".join([f"File: {filename}\n```\n{content}\n```" for filename, content in file_contents.items()])
    prompt = '''
    [Previous prompt content remains unchanged...]
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

def get_clipboard_content():
    """
    Get content directly from clipboard without waiting.
    """
    if pyperclip is None:
        print("Error: Clipboard functionality is not available.")
        sys.exit(1)
    try:
        content = pyperclip.paste()
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        while '\n\n\n' in content:
            content = content.replace('\n\n\n', '\n\n')
        return content.strip()
    except Exception as e:
        print(f"Error accessing clipboard: {e}")
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

class VoiceAssistant:
    def __init__(self):
        self.client = OpenAI()       
        self.recording = False
        self.auto_mode = False
        self.audio_queue = queue.Queue()
        self.ws = None
        self.running = True
        self.client = OpenAI()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create output text area
        self.output_text = scrolledtext.ScrolledText(self.main_frame, height=20)
        self.output_text.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create voice control button
        self.voice_button = ttk.Button(
            self.main_frame,
            text="🎤 Start Voice Control",
            command=self.toggle_voice_control
        )
        self.voice_button.grid(row=1, column=0, pady=10)
        
        # Create status label
        self.status_label = ttk.Label(self.main_frame, text="Ready")
        self.status_label.grid(row=1, column=1, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Start asyncio event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()

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
        self.recording = True
        self.voice_button.configure(text="🔴 Stop Voice Control")
        self.status_label.configure(text="Listening...")
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            callback=self.audio_callback,
            blocksize=CHUNK_SIZE
        )
        self.stream.start()
        
        # Connect to OpenAI WebSocket
        asyncio.run_coroutine_threadsafe(self.connect_websocket(), self.loop)

    def stop_voice_control(self):
        """Stop voice control"""
        self.recording = False
        self.voice_button.configure(text="🎤 Start Voice Control")
        self.status_label.configure(text="Ready")
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Close WebSocket connection
        if self.ws:
            asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            self.ws = None

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            self.log_message(f"Audio input error: {status}")
            return
        if self.recording:
            # Convert to 16-bit PCM
            audio_data = (indata * 32767).astype(np.int16)
            self.audio_queue.put(audio_data.tobytes())

    async def connect_websocket(self):
        """Connect to OpenAI's realtime websocket API"""
        try:
            self.ws = await websockets.connect(
                f"{OPENAI_WEBSOCKET_URL}?model=gpt-4o-realtime-preview-2024-10-01",
                extra_headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            
            # Initialize session
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "input_audio_transcription": {
                        "language": "en",
                        "model": "whisper-1"
                    },
                    "voice": "alloy",
                    "instructions": """
                    You are an AI assistant that helps control the aider code assistant through voice commands.
                    Commands you understand:
                    - Run aider with clipboard content
                    - Add files to aider (from current directory)
                    - Check for issues and send to aider
                    - Summarize what happened when aider finishes
                    
                    Always confirm what action you're taking and provide clear feedback.
                    """
                }
            }))
            
            self.log_message("Connected to OpenAI realtime API")
            
            # Start message handling
            asyncio.create_task(self.handle_websocket_messages())
            asyncio.create_task(self.process_audio_queue())
            
        except Exception as e:
            self.log_message(f"Failed to connect to OpenAI: {e}")
            self.stop_voice_control()

    async def handle_websocket_messages(self):
        """Handle incoming websocket messages"""
        while self.ws and self.recording:
            try:
                message = await self.ws.recv()
                event = json.loads(message)
                
                if event["type"] == "conversation.item.input_audio_transcription.completed":
                    text = event["transcription"]["text"]
                    self.log_message(f"You said: {text}")
                    await self.process_voice_command(text)
                    
                elif event["type"] == "error":
                    self.log_message(f"Error from OpenAI: {event['error']['message']}")
                    
            except websockets.exceptions.ConnectionClosed:
                self.log_message("WebSocket connection closed")
                break
            except Exception as e:
                self.log_message(f"Error handling websocket message: {e}")

    async def process_audio_queue(self):
        """Process audio queue and send to OpenAI"""
        while self.recording:
            if not self.audio_queue.empty():
                audio_data = b''
                while not self.audio_queue.empty():
                    audio_data += self.audio_queue.get()
                
                if self.ws:
                    try:
                        await self.ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio_data).decode()
                        }))
                    except Exception as e:
                        self.log_message(f"Error sending audio data: {e}")
            
            await asyncio.sleep(0.1)

    async def process_voice_command(self, text):
        """Process transcribed voice commands"""
        self.log_message(f"Processing command: {text}")
        
        if "run aider" in text.lower() and "clipboard" in text.lower():
            self.log_message("Running aider with clipboard content...")
            subprocess.Popen(
                ["python", "aider_wrapper.py", "-c"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            await self.send_audio_response("Started aider with clipboard content")
            
        elif "add files" in text.lower():
            self.log_message("Adding files to aider...")
            files = [f for f in os.listdir('.') if f.endswith(('.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx'))]
            if files:
                subprocess.Popen(
                    ["python", "aider_wrapper.py"] + files,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                await self.send_audio_response(f"Added {len(files)} files to aider")
            else:
                await self.send_audio_response("No code files found in current directory")
            
        elif "check" in text.lower() and "issues" in text.lower():
            self.log_message("Checking for issues...")
            await self.check_for_issues()
            
        else:
            await self.send_audio_response(
                "I didn't understand that command. You can say:\n" +
                "- Run aider with clipboard content\n" +
                "- Add files to aider\n" +
                "- Check for issues"
            )

    async def check_for_issues(self):
        """Check for code issues and send to aider"""
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
                with open(issues_file.name, 'w') as f:
                    f.write("\n\n".join(combined_issues))
                
                subprocess.Popen(
                    ["python", "aider_wrapper.py", "-i", issues_file.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                await self.send_audio_response("Found issues and sent them to aider for fixing")
                
                # Clean up temp file after a delay
                self.root.after(5000, lambda: os.unlink(issues_file.name))
            else:
                await self.send_audio_response("No issues found in the code")
                
        except Exception as e:
            self.log_message(f"Error checking for issues: {e}")
            await self.send_audio_response(f"Error checking for issues: {str(e)}")
        finally:
            self.fixing_issues = False
            # Cleanup temp files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
            self.temp_files = []

    async def monitor_aider_process(self, recheck=False):
        """Monitor aider process and handle completion"""
        if not self.aider_process:
            return

        try:
            # Create a task to read output
            async def read_output():
                while True:
                    output = await asyncio.get_event_loop().run_in_executor(
                        None, self.aider_process.stdout.readline
                    )
                    if not output:
                        break
                    print(output.decode().strip())

            # Start reading output
            asyncio.create_task(read_output())

            # Wait for process to complete
            await asyncio.get_event_loop().run_in_executor(
                None, self.aider_process.wait
            )

            # Process completed
            if self.aider_process.returncode == 0:
                await self.send_audio_response("Aider has finished successfully")
                if recheck:
                    print("Waiting for filesystem to settle...")
                    await asyncio.sleep(10)  # Increased wait time for filesystem changes
                    await self.recheck_issues()
            else:
                await self.send_audio_response("Aider encountered an error")

        except Exception as e:
            print(f"Error monitoring aider process: {e}")
            await self.send_audio_response(f"Error monitoring aider: {str(e)}")
        finally:
            self.aider_process = None

    async def recheck_issues(self):
        """Recheck for issues after aider fix attempt"""
        print("Rechecking for issues...")
        await self.send_audio_response("Checking if all issues were fixed...")

        issues_remain = False
        remaining_issues = []
        
        # Recheck ruff
        ruff_result = subprocess.run(
            ["ruff", "check", "."], 
            capture_output=True, 
            text=True
        )
        if ruff_result.stdout:
            issues_remain = True
            remaining_issues.append("Ruff issues still remain")
            print("Ruff issues still remain")

        # Wait before mypy check
        await asyncio.sleep(2)
        
        # Recheck mypy
        mypy_result = subprocess.run(
            ["mypy", "."], 
            capture_output=True, 
            text=True
        )
        if mypy_result.stdout:
            issues_remain = True
            remaining_issues.append("Mypy issues still remain")
            print("Mypy issues still remain")

        if issues_remain:
            message = "Some issues still remain: " + ", ".join(remaining_issues)
            await self.send_audio_response(message + ". Would you like me to try fixing them again?")
        else:
            await self.send_audio_response("All issues have been fixed successfully!")

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
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()
    async def handle_websocket_messages(self):
        """Handle incoming websocket messages"""
        while self.running:
            try:
                message = await self.ws.recv()
                event = json.loads(message)
                
                if event["type"] == "conversation.item.input_audio_transcription.completed":
                    text = event["transcription"]["text"]
                    self.last_transcription = text
                    await self.process_voice_command(text)
                    
                elif event["type"] == "error":
                    print(f"Error from OpenAI: {event['error']['message']}")
                    
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed")
                self.running = False
                break
            except Exception as e:
                print(f"Error handling websocket message: {e}")

    async def start_voice_interaction(self):
        """Start voice interaction loop"""
        await self.connect_websocket()
        
        try:
            with sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                callback=self.audio_callback,
                blocksize=CHUNK_SIZE
            ):
                print("Voice assistant is ready! Speak a command...")
                self.recording = True
                
                # Start websocket message handler
                asyncio.create_task(self.handle_websocket_messages())
                
                while self.running:
                    if not self.audio_queue.empty():
                        audio_data = b''
                        while not self.audio_queue.empty():
                            audio_data += self.audio_queue.get()
                        
                        # Send audio data to OpenAI
                        if self.ws:
                            try:
                                await self.ws.send(json.dumps({
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(audio_data).decode()
                                }))
                            except Exception as e:
                                print(f"Error sending audio data: {e}")
                    
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"Error in voice interaction: {e}")
        finally:
            self.recording = False
            if self.ws:
                await self.ws.close()
            # Cleanup any remaining temp files
            for temp_file in self.temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
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
    </task_description>

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

def get_clipboard_content():
    """
    Get content directly from clipboard without waiting.
    """
    if pyperclip is None:
        print("Error: Clipboard functionality is not available.")
        sys.exit(1)
    try:
        content = pyperclip.paste()
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        while '\n\n\n' in content:
            content = content.replace('\n\n\n', '\n\n')
        return content.strip()
    except Exception as e:
        print(f"Error accessing clipboard: {e}")
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

    if args.voice_only:
        assistant = VoiceAssistant()
        asyncio.run(assistant.start_voice_interaction())
        return

    if args.gui:
        root = tk.Tk()
        app = AiderVoiceGUI(root)
        if args.auto:
            app.auto_mode = True
            print("Auto mode enabled - will automatically send ruff issues to aider")
        root.mainloop()
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
