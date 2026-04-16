"""Synthetic-audio calibration harness for the stress-score heuristic.

This is *calibration* evidence, not empirical validation against market
outcomes. It generates audio with known acoustic properties and asserts
the stress score responds in the expected direction:

    1. Smooth 220 Hz sine           → very low stress (confident baseline)
    2. Pitch-jittered tone          → higher jitter + higher score
    3. Speech with frequent pauses  → higher hesitation + higher score
    4. Worst-case combined          → highest score

A real empirical validation would require a labeled dataset of earnings
calls tagged with subsequent price moves. The maintainer does not have
such a dataset. If you build one, please contribute it — see
CONTRIBUTING.md for scope.

Usage:
    python scripts/validate_stress_score.py

Exits 0 if all directional invariants hold, 1 otherwise.
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from live_audio_intelligence_mcp.prosody_analyzer import (
    StressResult,
    analyze_vocal_stress,
)


SAMPLE_RATE = 16000
DURATION = 3.0  # seconds per test clip


def _write_wav(path: Path, samples: np.ndarray) -> Path:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return path


def _smooth_sine(freq: float = 220.0, duration: float = DURATION) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _jittered_tone(duration: float = DURATION, seed: int = 0) -> np.ndarray:
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(seed=seed)
    fm = 230.0 + 50.0 * np.sin(2 * np.pi * 4.0 * t) + rng.normal(0, 5.0, n)
    phase = 2 * np.pi * np.cumsum(fm) / SAMPLE_RATE
    return (0.3 * np.sin(phase)).astype(np.float32)


def _speech_with_pauses(
    speech_ms: int = 500,
    pause_ms: int = 600,
    cycles: int = 3,
    freq: float = 220.0,
) -> np.ndarray:
    t_speech = np.linspace(
        0, speech_ms / 1000, int(SAMPLE_RATE * speech_ms / 1000), endpoint=False,
    )
    speech = (0.3 * np.sin(2 * np.pi * freq * t_speech)).astype(np.float32)
    pause = np.zeros(int(SAMPLE_RATE * pause_ms / 1000), dtype=np.float32)
    return np.tile(np.concatenate([speech, pause]), cycles)


def _combined_worst_case() -> np.ndarray:
    jittered = _jittered_tone(duration=0.5, seed=1)
    pause = np.zeros(int(SAMPLE_RATE * 0.6), dtype=np.float32)
    return np.tile(np.concatenate([jittered, pause]), 3)


def _fmt(label: str, r: StressResult) -> str:
    return (
        f"{label:<28} score={r.stress_score:5.1f}  "
        f"jitter={r.pitch_jitter:.4f}  hes={r.hesitation_ratio:.3f}  "
        f"voiced={r.voiced_fraction:.3f}  pauses={r.pause_count}"
    )


def main() -> int:
    print("Synthetic-audio calibration harness for live-audio-intelligence-mcp\n")
    print("This validates *directional* behavior of the stress heuristic.")
    print("It is NOT a substitute for empirical validation against labeled")
    print("market outcomes.\n")

    with tempfile.TemporaryDirectory(prefix="lai_calib_") as tdir:
        tdir_path = Path(tdir)

        fixtures = {
            "smooth_sine_220hz":     _smooth_sine(),
            "pitch_jittered_tone":   _jittered_tone(),
            "speech_with_pauses":    _speech_with_pauses(),
            "combined_worst_case":   _combined_worst_case(),
        }

        results: dict[str, StressResult] = {}
        for name, samples in fixtures.items():
            path = _write_wav(tdir_path / f"{name}.wav", samples)
            results[name] = analyze_vocal_stress(path)
            print(_fmt(name, results[name]))

        print()

        checks: list[tuple[str, bool]] = []

        # 1. Smooth sine should score in the "confident" band (< 20).
        checks.append((
            "smooth sine scores < 20 (confident baseline)",
            results["smooth_sine_220hz"].stress_score < 20.0,
        ))

        # 2. Jittered tone must have strictly higher jitter than smooth.
        checks.append((
            "jittered tone has higher jitter than smooth sine",
            results["pitch_jittered_tone"].pitch_jitter
            > results["smooth_sine_220hz"].pitch_jitter,
        ))

        # 3. Jittered tone scores strictly higher than smooth.
        checks.append((
            "jittered tone scores higher than smooth sine",
            results["pitch_jittered_tone"].stress_score
            > results["smooth_sine_220hz"].stress_score,
        ))

        # 4. Pause-padded speech has meaningful hesitation_ratio.
        checks.append((
            "pause-padded speech has hesitation_ratio > 0.15",
            results["speech_with_pauses"].hesitation_ratio > 0.15,
        ))

        # 5. Pause-padded scores higher than smooth on hesitation contribution.
        checks.append((
            "pause-padded scores higher than smooth sine",
            results["speech_with_pauses"].stress_score
            > results["smooth_sine_220hz"].stress_score,
        ))

        # 6. Combined worst-case scores the highest of all fixtures.
        worst = results["combined_worst_case"].stress_score
        others = [r.stress_score for k, r in results.items() if k != "combined_worst_case"]
        checks.append((
            "combined worst-case scores highest of all fixtures",
            worst >= max(others),
        ))

        # 7. All scores stay bounded to [0, 100].
        checks.append((
            "all scores within [0, 100]",
            all(0.0 <= r.stress_score <= 100.0 for r in results.values()),
        ))

        print("Directional checks:")
        all_ok = True
        for label, ok in checks:
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_ok = False
            print(f"  [{tag}] {label}")

        print()
        if all_ok:
            print("All directional checks passed.")
            return 0
        print("One or more directional checks failed — heuristic has regressed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
