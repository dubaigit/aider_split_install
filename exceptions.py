"""Custom exceptions for the Aider Voice Assistant."""

class AiderError(Exception):
    """Base exception class for Aider Voice Assistant."""
    pass

class AudioError(AiderError):
    """Raised when there are audio processing issues."""
    pass

class WebSocketError(AiderError):
    """Base class for WebSocket related errors."""
    pass

class ConnectionError(WebSocketError):
    """Raised when there are connection issues."""
    pass

class AuthenticationError(WebSocketError):
    """Raised when authentication fails."""
    pass

class StateError(AiderError):
    """Raised when invalid state transitions occur."""
    pass

class ConfigurationError(AiderError):
    """Raised when there are configuration issues."""
    pass

class ValidationError(AiderError):
    """Raised when validation fails."""
    pass
"""Custom exceptions for the Aider Voice Assistant."""

class AiderError(Exception):
    """Base exception for all Aider-related errors."""
    pass

class AudioError(AiderError):
    """Base exception for audio-related errors."""
    pass

class AudioProcessingError(AudioError):
    """Exception for audio processing errors."""
    pass

class AudioDeviceError(AudioError):
    """Exception for audio device errors."""
    pass

class WebSocketError(AiderError):
    """Base exception for WebSocket-related errors."""
    def __init__(self, message, original_error=None):
        super().__init__(message)
        self.original_error = original_error

class WebSocketConnectionError(WebSocketError):
    """Exception for WebSocket connection errors."""
    pass

class WebSocketTimeoutError(WebSocketError):
    """Exception for WebSocket timeout errors."""
    pass

class WebSocketAuthenticationError(WebSocketError):
    """Exception for WebSocket authentication errors."""
    pass

class StateError(AiderError):
    """Exception for state-related errors."""
    pass

class ValidationError(AiderError):
    """Exception for validation errors."""
    pass

class ConfigurationError(AiderError):
    """Exception for configuration-related errors."""
    pass
