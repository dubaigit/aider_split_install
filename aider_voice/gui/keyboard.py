"""Keyboard shortcut handling."""

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
