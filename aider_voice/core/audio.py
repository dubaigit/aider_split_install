"""Audio processing components for Aider Voice Assistant."""

import base64
import json
import pyaudio
import time
from queue import Empty as QueueEmpty
from .exceptions import AudioProcessingError

CHUNK_SIZE = 1024
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
REENGAGE_DELAY_MS = 500

class AudioBufferManager:
    """Manages audio buffering and processing"""

    def __init__(self, max_size, chunk_size, sample_rate):
        self.max_size = max_size
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.stats = {"drops": 0, "overflows": 0}

    def get_chunks(self, queue):
        """Get chunks from queue with overflow protection"""
        chunks = []
        while len(self.buffer) < self.max_size:
            try:
                chunk = queue.get_nowait()
                if len(self.buffer) + len(chunk) <= self.max_size:
                    chunks.append(chunk)
                else:
                    self.stats["overflows"] += 1
                    break
            except QueueEmpty:
                break
        return chunks

    def combine_chunks(self, chunks):
        """Combine chunks with error checking"""
        try:
            return b"".join(chunks)
        except Exception as e:
            self.stats["drops"] += 1
            raise AudioProcessingError(f"Error combining chunks: {e}") from e

    def get_usage(self):
        """Get current buffer usage ratio"""
        return len(self.buffer) / self.max_size
