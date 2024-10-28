"""Tests for the Aider Voice Assistant."""

import asyncio
import json
import os
import time
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pyperclip
import websockets

from aider_wrapper import (AiderVoiceGUI, AsyncTestCase, ClipboardManager,
                         ConnectionState, VoiceCommandProcessor, WebSocketManager)


class TestVoiceCommandProcessor(AsyncTestCase):
    """Test suite for VoiceCommandProcessor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        self.parent = MagicMock()
        self.processor = VoiceCommandProcessor(self.parent)

    def test_validate_command_empty(self):
        """Test validation of empty commands"""
        self.assertFalse(self.processor.validate_command(""))
        self.assertFalse(self.processor.validate_command(None))
        self.assertFalse(self.processor.validate_command("   "))

    def test_validate_command_length(self):
        """Test validation of command length"""
        # Test command that exceeds max length
        long_command = "a" * 1001
        self.assertFalse(self.processor.validate_command(long_command),
                        "Should reject commands longer than 1000 chars")
        
        # Test command at max length
        valid_command = "a" * 1000
        self.assertTrue(self.processor.validate_command(valid_command),
                       "Should accept commands at max length")
        
        # Test normal length command
        self.assertTrue(self.processor.validate_command("normal command"),
                       "Should accept normal length commands")

    def test_validate_command_profanity(self):
        """Test validation of command content"""
        # Test commands with profanity
        self.assertFalse(self.processor.validate_command("profanity1 test"),
                        "Should reject commands with profanity1")
        self.assertFalse(self.processor.validate_command("test profanity2"),
                        "Should reject commands with profanity2")
        
        # Test normal commands
        self.assertTrue(self.processor.validate_command("normal command"),
                       "Should accept normal commands")
        self.assertTrue(self.processor.validate_command("hello world"),
                       "Should accept greetings")
        self.assertTrue(self.processor.validate_command("test case"),
                       "Should accept test commands")


class TestArgumentParsing(unittest.TestCase):
    """Test command line argument parsing"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.original_parser = AiderVoiceGUI._parser if hasattr(AiderVoiceGUI, '_parser') else None

    def setUp(self):
        """Set up test environment for each test"""
        self.parser_patcher = patch('argparse.ArgumentParser')
        self.mock_parser = self.parser_patcher.start()
        self.mock_args = MagicMock()
        self.mock_parser.return_value.parse_args.return_value = self.mock_args
        
        # Save any existing error handlers
        self.original_error_handler = self.mock_parser.return_value.error \
            if hasattr(self.mock_parser.return_value, 'error') else None

    def tearDown(self):
        """Clean up test environment after each test"""
        self.parser_patcher.stop()
        
        # Restore original error handler if any
        if self.original_error_handler:
            self.mock_parser.return_value.error = self.original_error_handler

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test environment"""
        if cls.original_parser:
            AiderVoiceGUI._parser = cls.original_parser

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
        super().setUp()
        self.root = tk.Tk()
        self.app = AiderVoiceGUI(self.root)

    def tearDown(self):
        """Clean up test environment"""
        super().tearDown()
        self.root.destroy()

    def test_keyboard_shortcuts(self):
        """Test keyboard shortcut bindings and handlers"""
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
                # Simulate key press
                self.app.root.event_generate(key)
                self.root.update_idletasks()
                self.root.update()
                
                # Verify method was called
                mock_method.assert_called_once_with(),\
                    f"Shortcut {key} should trigger {method_name}"
                mock_method.reset_mock()

        # Test invalid shortcut doesn't trigger anything
        with patch.object(self.app, 'log_message') as mock_log:
            self.app.root.event_generate('<Control-x>')
            self.root.update_idletasks()
            self.root.update()
            mock_log.assert_not_called(),\
                "Invalid shortcut should not trigger any logging"


if __name__ == '__main__':
    unittest.main()
