"""Syntactic URL validation — no network, no subprocess."""

from __future__ import annotations

import pytest

from live_audio_intelligence_mcp.audio_streamer import StreamManager
from live_audio_intelligence_mcp.exceptions import InvalidStreamURLError


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/live.m3u8",
        "http://example.com/stream",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://cnbc.com/live",
    ],
)
def test_accepts_http_and_https(url):
    # Should not raise — semantic check against yt-dlp happens later.
    StreamManager._validate_url_syntax(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/plain;base64,SGVsbG8=",
        "ftp://example.com/stream",
        "gopher://example.com",
        "ssh://user@host/path",
    ],
)
def test_rejects_non_http_schemes(url):
    with pytest.raises(InvalidStreamURLError):
        StreamManager._validate_url_syntax(url)


@pytest.mark.parametrize("url", ["", "   ", "\n\t"])
def test_rejects_empty_or_whitespace(url):
    with pytest.raises(InvalidStreamURLError):
        StreamManager._validate_url_syntax(url)


def test_rejects_non_string():
    with pytest.raises(InvalidStreamURLError):
        StreamManager._validate_url_syntax(None)  # type: ignore[arg-type]


def test_rejects_hostless():
    with pytest.raises(InvalidStreamURLError):
        StreamManager._validate_url_syntax("http:///path-only")


def test_invalid_url_error_is_also_value_error():
    # Belt-and-braces: confirms the exception hierarchy integrates with
    # the URL validator the way clients catching ValueError expect.
    with pytest.raises(ValueError):
        StreamManager._validate_url_syntax("file:///etc/shadow")
