"""Exception hierarchy must preserve backward-compat with ValueError / RuntimeError."""

from __future__ import annotations

import pytest

from live_audio_intelligence_mcp.exceptions import (
    InvalidStreamURLError,
    LiveAudioError,
    StreamLimitExceededError,
    StreamStartError,
    UnknownStreamError,
)


def test_base_is_exception():
    assert issubclass(LiveAudioError, Exception)


@pytest.mark.parametrize("cls", [InvalidStreamURLError, UnknownStreamError])
def test_value_error_subclasses(cls):
    # Clients that catch ValueError for bad-input must still catch these.
    assert issubclass(cls, ValueError)
    assert issubclass(cls, LiveAudioError)


@pytest.mark.parametrize("cls", [StreamLimitExceededError, StreamStartError])
def test_runtime_error_subclasses(cls):
    assert issubclass(cls, RuntimeError)
    assert issubclass(cls, LiveAudioError)


def test_raise_and_catch_as_value_error():
    with pytest.raises(ValueError):
        raise InvalidStreamURLError("bad url")


def test_raise_and_catch_as_runtime_error():
    with pytest.raises(RuntimeError):
        raise StreamLimitExceededError("too many")


def test_message_roundtrip():
    exc = UnknownStreamError("stream abc123 not active")
    assert "abc123" in str(exc)
