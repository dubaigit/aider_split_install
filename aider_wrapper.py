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

class VoiceAssistant:
    def __init__(self):
        self.client = OpenAI()
        self.recording = False
        self.audio_queue = queue.Queue()
        self.ws = None
        self.running = True
        self.current_command = None
        self.last_transcription = None

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
            # Initialize session with voice settings
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "input_audio_transcription": True,
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
            print("Connected to OpenAI realtime API")
        except Exception as e:
            print(f"Failed to connect to OpenAI: {e}")
            self.running = False

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio input error: {status}")
            return
        if self.recording:
            # Convert to 16-bit PCM
            audio_data = (indata * 32767).astype(np.int16)
            self.audio_queue.put(audio_data.tobytes())

    async def process_voice_command(self, text):
        """Process transcribed voice commands"""
        print(f"Processing command: {text}")
        
        if "run aider" in text.lower() and "clipboard" in text.lower():
            print("Running aider with clipboard content...")
            self.run_aider_with_clipboard()
            
        elif "add files" in text.lower():
            print("Adding files to aider...")
            await self.add_files_to_aider()
            
        elif "check" in text.lower() and "issues" in text.lower():
            print("Checking for issues...")
            await self.check_for_issues()
            
        else:
            print("Command not recognized")
            await self.send_audio_response("I didn't understand that command. You can ask me to run aider with clipboard content, add files, or check for issues.")

    def run_aider_with_clipboard(self):
        """Run aider using clipboard content"""
        try:
            clipboard_content = pyperclip.paste()
            subprocess.Popen(["python", "aider_wrapper.py", "-c"], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
            print("Started aider with clipboard content")
        except Exception as e:
            print(f"Error running aider with clipboard: {e}")

    async def add_files_to_aider(self):
        """Add files from current directory to aider"""
        try:
            files = os.listdir('.')
            code_files = [f for f in files if f.endswith(('.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx'))]
            if code_files:
                subprocess.Popen(["python", "aider_wrapper.py"] + code_files,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
                await self.send_audio_response(f"Added {len(code_files)} files to aider")
            else:
                await self.send_audio_response("No code files found in current directory")
        except Exception as e:
            print(f"Error adding files to aider: {e}")

    async def check_for_issues(self):
        """Check for code issues and send to aider"""
        try:
            # Run ruff for Python files
            result = subprocess.run(["ruff", "check", "."], 
                                  capture_output=True, 
                                  text=True)
            if result.stdout:
                print("Issues found, sending to aider...")
                with open("issues.txt", "w") as f:
                    f.write(result.stdout)
                subprocess.Popen(["python", "aider_wrapper.py", "-i", "issues.txt"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
                await self.send_audio_response("Found issues and sent them to aider for fixing")
            else:
                await self.send_audio_response("No issues found in the code")
        except Exception as e:
            print(f"Error checking for issues: {e}")

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
                print(f"Error sending audio response: {e}")

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
            with sd.InputStream(channels=CHANNELS,
                              samplerate=SAMPLE_RATE,
                              callback=self.audio_callback,
                              blocksize=CHUNK_SIZE):
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
            # Cleanup handled by tempfile module
            pass

def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()

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
    args = parser.parse_args()

    instructions = None
    
    if args.voice_only:
        assistant = VoiceAssistant()
        asyncio.run(assistant.start_voice_interaction())
        return
    else:
        print("Warning: Git functionality is disabled. Skipping git commit.")

    # Get instructions content
    if args.clipboard:
        instructions = get_clipboard_content()
    elif hasattr(args, 'instructions') and args.instructions:
        instructions = read_file_content(args.instructions)
    else:
        print("Error: No instructions provided")
        sys.exit(1)

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
        process = subprocess.Popen(aider_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True, bufsize=1)

        handle_aider_prompts(process)

        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, aider_command)
    except subprocess.CalledProcessError as e:
        print(f"Error executing aider command: {e}", file=sys.stderr)
        print("The specified model may not be supported or there might be an issue with the aider configuration.")
        print("Please check your aider installation and ensure the model is correctly specified.")
        print("You can try running the aider command directly to see more detailed error messages.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        print("Please report this issue to the developers.")
        sys.exit(1)

    try:
        new_file_path = os.path.join(os.getcwd(), os.path.basename(temp_message_file.name))
        shutil.move(temp_message_file.name, new_file_path)
        print(f"\nTemporary file moved to: {new_file_path}")
    except IOError as e:
        print(f"Error moving temporary file: {e}")
    finally:
        try:
            os.unlink(temp_message_file.name)
        except OSError:
            pass

if __name__ == "__main__":
    main()
