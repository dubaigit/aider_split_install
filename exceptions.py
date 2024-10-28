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
