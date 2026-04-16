"""Shared pytest fixtures.

Keep fixtures fast and pure: no ffmpeg, no network, no Whisper model.
The tests in this directory must run green on a laptop with only the
package + its Python dependencies installed.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest


SAMPLE_RATE = 16000


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write a mono int16 WAV file."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


@pytest.fixture
def write_wav(tmp_path: Path):
    """Return a function that writes mono 16kHz WAV into the tmp dir."""

    def _make(name: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
        return _write_wav(tmp_path / name, samples, sample_rate)

    return _make


@pytest.fixture
def smooth_sine() -> np.ndarray:
    """3-second 220 Hz sine tone — no jitter, no pauses. Should score low."""
    duration = 3.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def jittered_sine() -> np.ndarray:
    """3-second sine with large pitch wobble — should score higher on jitter."""
    duration = 3.0
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Slow amplitude-modulated pitch drift between ~180 Hz and ~280 Hz,
    # plus a small noise term in the instantaneous frequency.
    rng = np.random.default_rng(seed=0)
    fm = 230.0 + 50.0 * np.sin(2 * np.pi * 4.0 * t) + rng.normal(0, 5.0, n)
    phase = 2 * np.pi * np.cumsum(fm) / SAMPLE_RATE
    return (0.3 * np.sin(phase)).astype(np.float32)


@pytest.fixture
def silence() -> np.ndarray:
    """3 seconds of near-silence — triggers the 'insufficient voiced frames' path."""
    return np.zeros(int(SAMPLE_RATE * 3.0), dtype=np.float32)


@pytest.fixture
def sine_with_long_pauses() -> np.ndarray:
    """Sine + injected 600ms silence gaps — should score high on hesitation."""
    t_speech = np.linspace(0, 0.5, int(SAMPLE_RATE * 0.5), endpoint=False)
    speech = (0.3 * np.sin(2 * np.pi * 220 * t_speech)).astype(np.float32)
    pause = np.zeros(int(SAMPLE_RATE * 0.6), dtype=np.float32)
    # 0.5s speech, 0.6s pause, repeated → hesitation_ratio ≈ 0.6 / 1.1 ≈ 0.55.
    return np.tile(np.concatenate([speech, pause]), 3)
