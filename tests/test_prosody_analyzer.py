"""Prosody analyzer directional checks on synthetic audio.

These tests don't assert exact numeric scores (the heuristic is allowed to
be retuned) — they assert that the score responds in the expected direction
to controlled acoustic manipulations. If these break, the heuristic has
lost its discriminative power.
"""

from __future__ import annotations

import numpy as np
import pytest

from live_audio_intelligence_mcp.prosody_analyzer import (
    StressResult,
    _empty_result,
    analyze_multiple_chunks,
    analyze_vocal_stress,
)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_vocal_stress(tmp_path / "does-not-exist.wav")


def test_empty_result_carries_reason():
    r = _empty_result("because I said so")
    assert r.stress_score == 0.0
    assert "because" in r.analysis


def test_to_dict_has_expected_keys(smooth_sine, write_wav):
    path = write_wav("tone.wav", smooth_sine)
    d = analyze_vocal_stress(path).to_dict()
    expected = {
        "stress_score", "pitch_mean_hz", "pitch_std_hz", "pitch_jitter",
        "hesitation_ratio", "voiced_fraction", "pause_count",
        "longest_pause_ms", "analysis",
    }
    assert expected <= set(d.keys())


def test_silence_returns_empty_result(silence, write_wav):
    path = write_wav("silence.wav", silence)
    r = analyze_vocal_stress(path)
    # Pure silence should fall into the "insufficient voiced frames" branch
    # (or, failing that, at least produce a very low voiced fraction).
    assert r.stress_score == 0.0 or r.voiced_fraction < 0.1


def test_smooth_sine_has_low_jitter(smooth_sine, write_wav):
    path = write_wav("smooth.wav", smooth_sine)
    r = analyze_vocal_stress(path)
    # A clean 220 Hz sine has essentially zero F0 variance. If pyin
    # tracks it and reports CV > 0.05 something is badly wrong.
    assert r.pitch_jitter < 0.05
    # Mean pitch should land near 220 Hz (pyin quantises to a musical grid
    # so allow ±25 Hz slack).
    assert 180 < r.pitch_mean_hz < 260


def test_jittered_sine_has_higher_jitter(smooth_sine, jittered_sine, write_wav):
    """The jittered-FM signal must have strictly higher jitter than a flat tone."""
    smooth = analyze_vocal_stress(write_wav("smooth.wav", smooth_sine))
    jittered = analyze_vocal_stress(write_wav("jittered.wav", jittered_sine))
    assert jittered.pitch_jitter > smooth.pitch_jitter
    # Directional stress-score check: more jitter → higher stress.
    assert jittered.stress_score > smooth.stress_score


def test_long_pauses_raise_hesitation(sine_with_long_pauses, write_wav):
    path = write_wav("pauses.wav", sine_with_long_pauses)
    r = analyze_vocal_stress(path)
    # Multiple 600ms silence gaps → pause_count ≥ 2 and a non-trivial
    # hesitation_ratio. Exact values depend on librosa.effects.split's
    # top_db gating, so we assert ranges, not point values.
    assert r.pause_count >= 2
    assert r.hesitation_ratio > 0.15


def test_score_is_bounded(jittered_sine, sine_with_long_pauses, write_wav):
    """Score must stay clamped to [0, 100] even for pathological inputs."""
    for name, samples in [("jittered", jittered_sine), ("pauses", sine_with_long_pauses)]:
        r = analyze_vocal_stress(write_wav(f"{name}.wav", samples))
        assert 0.0 <= r.stress_score <= 100.0


def test_analyze_multiple_chunks_aggregates(smooth_sine, write_wav):
    paths = [
        write_wav("a.wav", smooth_sine),
        write_wav("b.wav", smooth_sine),
        write_wav("c.wav", smooth_sine),
    ]
    r = analyze_multiple_chunks(paths)
    assert isinstance(r, StressResult)
    assert 0.0 <= r.stress_score <= 100.0


def test_analyze_multiple_chunks_no_valid_input(tmp_path):
    # All paths are invalid — function should return an _empty_result, not crash.
    r = analyze_multiple_chunks([tmp_path / "nope.wav"])
    assert r.stress_score == 0.0
    assert "No valid chunks" in r.analysis


def test_very_short_audio_returns_empty(write_wav):
    # 100ms of tone — below the 0.5s duration gate in analyze_vocal_stress.
    short = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.1, 1600, endpoint=False))).astype(np.float32)
    path = write_wav("short.wav", short)
    r = analyze_vocal_stress(path)
    assert r.stress_score == 0.0
    assert "too short" in r.analysis.lower()
