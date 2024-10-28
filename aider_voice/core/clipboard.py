"""Clipboard monitoring and management."""

import asyncio
import pyperclip

class ClipboardManager:
    """Manages clipboard monitoring and content processing"""

    def __init__(self, parent):
        self.parent = parent
        self.previous_content = ""
        self.monitoring = False
        self.monitoring_task = None
        self.update_interval = 0.5
        self.max_content_size = 1024 * 1024
        self.history = []
        self.interface_state = parent.interface_state
        self.log_message = parent.log_message
        self.processors = {
            "code": self.process_code,
            "text": self.process_text,
            "url": self.process_url,
        }

    # ... rest of ClipboardManager implementation ...
