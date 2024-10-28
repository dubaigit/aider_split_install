"""File management utilities."""

import os
import threading
import time

class FileManager:
    """Manages file operations and tracking"""
    
    def __init__(self, parent):
        self.parent = parent
        self.watched_files = {}
        self.file_hashes = {}
        self.last_modified = {}
        self._lock = threading.Lock()
        
    # ... rest of FileManager implementation ...
