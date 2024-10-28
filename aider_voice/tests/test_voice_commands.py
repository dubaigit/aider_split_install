"""Test suite for voice command processing."""

import unittest
from unittest.mock import patch, MagicMock
import asyncio
from contextlib import contextmanager

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
