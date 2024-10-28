"""Custom exceptions for the Aider Voice Assistant."""

class AiderError(Exception):
    """Base exception class for Aider Voice Assistant."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error
        self.timestamp = time.time()
        
    def __str__(self):
        base_msg = super().__str__()
        if self.original_error:
            return f"{base_msg} (Caused by: {type(self.original_error).__name__}: {str(self.original_error)})"
        return base_msg

class AudioError(AiderError):
    """Base class for audio-related errors."""
    pass

class AudioProcessingError(AudioError):
    """Error during audio processing."""
    pass

class AudioDeviceError(AudioError):
    """Error with audio input/output devices."""
    pass

class WebSocketError(AiderError):
    """Base class for WebSocket-related errors."""
    pass

class WebSocketConnectionError(WebSocketError):
    """Error establishing or maintaining WebSocket connection."""
    pass

class WebSocketTimeoutError(WebSocketError):
    """WebSocket operation timed out."""
    pass

class WebSocketAuthenticationError(WebSocketError):
    """WebSocket authentication failed."""
    pass

class StateError(AiderError):
    """Error in application state management."""
    pass

class ValidationError(AiderError):
    """Error in data validation."""
    pass

class ConfigurationError(AiderError):
    """Error in application configuration."""
    pass
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
