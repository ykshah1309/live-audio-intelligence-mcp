"""StreamManager bookkeeping — no real stream start, just state + cap checks."""

from __future__ import annotations

import threading

import pytest

from live_audio_intelligence_mcp.audio_streamer import StreamManager, StreamState
from live_audio_intelligence_mcp.exceptions import StreamLimitExceededError


def _fake_state(sid: str) -> StreamState:
    # Construct without going through yt-dlp / ffmpeg.
    from pathlib import Path
    return StreamState(
        stream_id=sid,
        url="https://example.com/fake",
        temp_dir=Path("."),
    )


def test_default_cap_is_four():
    sm = StreamManager()
    assert sm._max_concurrent == 4


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("LAI_MAX_CONCURRENT_STREAMS", "10")
    sm = StreamManager()
    assert sm._max_concurrent == 10


def test_env_var_junk_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LAI_MAX_CONCURRENT_STREAMS", "not-a-number")
    sm = StreamManager()
    assert sm._max_concurrent == StreamManager.DEFAULT_MAX_CONCURRENT_STREAMS


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("LAI_MAX_CONCURRENT_STREAMS", "10")
    sm = StreamManager(max_concurrent_streams=2)
    assert sm._max_concurrent == 2


def test_cap_never_less_than_one(monkeypatch):
    monkeypatch.setenv("LAI_MAX_CONCURRENT_STREAMS", "0")
    sm = StreamManager()
    assert sm._max_concurrent >= 1


def test_start_stream_enforces_cap(monkeypatch):
    """Fill the internal streams dict and confirm start_stream refuses new ones.

    We skip the real yt-dlp / ffmpeg path by pre-populating _streams so the
    cap check fires before any subprocess would be spawned.
    """
    sm = StreamManager(max_concurrent_streams=2)
    sm._streams["a"] = _fake_state("a")
    sm._streams["b"] = _fake_state("b")

    with pytest.raises(StreamLimitExceededError):
        sm.start_stream("https://example.com/another")


def test_active_streams_property():
    sm = StreamManager()
    sm._streams["x"] = _fake_state("x")
    sm._streams["y"] = _fake_state("y")
    assert set(sm.active_streams) == {"x", "y"}


def test_get_state_unknown_returns_none():
    sm = StreamManager()
    assert sm.get_state("does-not-exist") is None


def test_stop_stream_unknown_returns_false():
    sm = StreamManager()
    assert sm.stop_stream("does-not-exist") is False


def test_get_chunks_unknown_returns_empty():
    sm = StreamManager()
    assert sm.get_chunks("does-not-exist") == []


def test_lock_exists():
    # Sanity: the internal lock is a real threading lock. Prior bugs have
    # involved replacing it with something non-thread-safe.
    sm = StreamManager()
    assert isinstance(sm._lock, type(threading.Lock()))
