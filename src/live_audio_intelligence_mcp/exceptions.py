"""Custom exceptions for live-audio-intelligence-mcp.

Using specific exception types (instead of raw RuntimeError / ValueError) lets
MCP clients distinguish between "the caller made a bad request" and "something
broke inside the server" without parsing error strings.
"""

from __future__ import annotations


class LiveAudioError(Exception):
    """Base class for all errors raised by this package."""


class InvalidStreamURLError(LiveAudioError, ValueError):
    """Raised when a user-supplied URL cannot be used as an audio source.

    Subclasses ValueError so existing MCP clients that catch ValueError
    (e.g. for input-validation errors) continue to work.
    """


class StreamLimitExceededError(LiveAudioError, RuntimeError):
    """Raised when the configured max number of concurrent streams is reached."""


class StreamStartError(LiveAudioError, RuntimeError):
    """Raised when yt-dlp / ffmpeg fails to start for a stream."""


class UnknownStreamError(LiveAudioError, ValueError):
    """Raised when the caller references a stream_id that isn't active."""
