"""Test suite for voice command processing."""

import unittest
from unittest.mock import patch, MagicMock
import asyncio
from contextlib import contextmanager

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
        if len(command) > 1000:  # Prevent overly long commands
            return False
        if command.isspace():  # Reject whitespace-only commands
            return False
        # Basic profanity check
        profanity = {'profanity1', 'profanity2'}  # Add actual words as needed
        if any(word in command.lower() for word in profanity):
            return False
        return True

class AsyncTestCase(unittest.TestCase):
    """Base class for async tests with enhanced async support"""

    def setUp(self):
        """Set up test environment with proper async context"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.addCleanup(self.cleanup_loop)

    # ... rest of AsyncTestCase implementation ...

class TestVoiceCommandProcessor(AsyncTestCase):
    """Test suite for VoiceCommandProcessor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.parent = MagicMock()
        self.processor = VoiceCommandProcessor(self.parent)

    # ... rest of TestVoiceCommandProcessor implementation ...
