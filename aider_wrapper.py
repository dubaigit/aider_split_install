# [Previous imports remain unchanged...]

# Update the WebSocket URL to match the current API endpoint
OPENAI_WEBSOCKET_URL = "wss://api.openai.com/v1/audio/realtime"  # Updated endpoint

class AiderVoiceGUI:
    def __init__(self, root):
        self.root = root
        self.ws = None
        self.response_active = False
        self.last_transcript_id = None
        self.audio_buffer = bytearray()
        self.last_audio_time = time.time()
        
    async def connect_websocket(self):
        """Connect to OpenAI's realtime websocket API"""
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime-audio/v1"
            }
            
            self.ws = await websockets.connect(
                OPENAI_WEBSOCKET_URL,
                extra_headers=headers
            )
        
            # Initialize session with current API configuration
            await self.ws.send(json.dumps({
                "type": "init",
                "model": "gpt-4-1106-preview",
                "settings": {
                    "speech": {
                        "model": "tts-1",
                        "voice": "alloy"
                    },
                    "vad": {
                        "enabled": true,
                        "threshold": 0.5,
                        "min_silence_duration_ms": 300
                    },
                    "temperature": 0.8,
                    "max_tokens": 2048,
                    "system_prompt": """
                    You are an AI assistant that helps control the Aider code assistant through voice commands.
                    
                    When a file is added:
                    1. Acknowledge the file addition
                    2. Provide a summary of the file contents (imports, classes, functions)
                    3. Ask if the user wants to analyze for issues or make changes
                    
                    When checking for issues:
                    1. Summarize the number and types of issues found
                    2. Ask if the user wants to fix them using Aider
                    
                    Commands you understand:
                    - Run Aider with clipboard content
                    - Add files to Aider (from current directory)
                    - Check for issues and send to Aider
                    - Summarize what happened when Aider finishes
                    
                    Always confirm what action you're taking and provide clear feedback.
                    Your knowledge cutoff is 2023-10. Be helpful, witty, and friendly.
                    Talk quickly and be engaging with a lively tone.
                    """
                }
            }))
            
            self.log_message("Connected to OpenAI realtime API")
            
            # Initialize response state
            self.response_active = False
            self.last_transcript_id = None
            self.audio_buffer = bytearray()
            self.last_audio_time = time.time()
            
            # Start message handling
            asyncio.create_task(self.handle_websocket_messages())
            asyncio.create_task(self.process_audio_queue())
            
        except websockets.exceptions.InvalidStatusCode as e:
            self.log_message(f"Failed to connect to OpenAI API: HTTP {e.status_code}")
            if e.status_code == 404:
                self.log_message("Error: API endpoint not found. Please check if the API URL is correct.")
            elif e.status_code == 401:
                self.log_message("Error: Authentication failed. Please check your OpenAI API key.")
            elif e.status_code == 429:
                self.log_message("Error: Rate limit exceeded. Please try again later.")
            else:
                self.log_message(f"Error: Unexpected status code {e.status_code}")
            self.stop_voice_control()
        except Exception as e:
            self.log_message(f"Failed to connect to OpenAI: {e}")
            self.stop_voice_control()

    # [Rest of the code remains unchanged...]
