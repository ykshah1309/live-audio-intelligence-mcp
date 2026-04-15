"""
Vocal Prosody & Stress Analyzer — the alpha generator.

Extracts fundamental frequency (F0) via librosa.pyin, computes pitch jitter
(variance in pitch over time), and measures hesitation ratio (fraction of
audio occupied by pauses > 400ms). Together these produce a 0–100 stress
score that flags executive vocal distress on earnings calls.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def _log(msg: str) -> None:
    print(f"[ProsodyAnalyzer] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Constants tuned for human speech on earnings calls
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
F0_MIN = 65.0   # Hz — roughly C2, deep male voice floor
F0_MAX = 600.0  # Hz — high female voice ceiling
PAUSE_THRESHOLD_MS = 400  # silence longer than this = hesitation
SILENCE_TOP_DB = 30  # dB below peak to consider as silence


@dataclass
class StressResult:
    """Structured output returned to the MCP client."""

    stress_score: float  # 0–100 composite score
    pitch_mean_hz: float
    pitch_std_hz: float
    pitch_jitter: float  # normalised jitter (coefficient of variation)
    hesitation_ratio: float  # 0.0–1.0, fraction of audio that is silence >400ms
    voiced_fraction: float  # 0.0–1.0, how much of the audio is voiced
    pause_count: int  # number of pauses >400ms
    longest_pause_ms: float
    analysis: str  # human-readable interpretation

    def to_dict(self) -> dict:
        return {
            "stress_score": round(self.stress_score, 1),
            "pitch_mean_hz": round(self.pitch_mean_hz, 1),
            "pitch_std_hz": round(self.pitch_std_hz, 2),
            "pitch_jitter": round(self.pitch_jitter, 4),
            "hesitation_ratio": round(self.hesitation_ratio, 4),
            "voiced_fraction": round(self.voiced_fraction, 4),
            "pause_count": self.pause_count,
            "longest_pause_ms": round(self.longest_pause_ms, 1),
            "analysis": self.analysis,
        }


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_vocal_stress(audio_chunk_path: str | Path) -> StressResult:
    """Analyze a single audio chunk for vocal stress indicators.

    This is a blocking / CPU-bound call. The MCP server should invoke it
    via ``asyncio.to_thread()`` to avoid stalling the event loop.
    """
    path = Path(audio_chunk_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio chunk not found: {path}")

    # Load WAV directly via soundfile — bypasses librosa.load's audioread
    # fallback which on some platforms spawns a child ffmpeg that inherits
    # our stdin (the MCP JSON-RPC pipe) and hangs.
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    duration_s = len(y) / sr

    if duration_s < 0.5:
        return _empty_result("Audio too short for analysis")

    # ------------------------------------------------------------------
    # 1. Fundamental Frequency via pYIN
    # ------------------------------------------------------------------
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        sr=sr,
        fmin=F0_MIN,
        fmax=F0_MAX,
        frame_length=2048,
    )

    # Filter to voiced frames only
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    voiced_fraction = len(voiced_f0) / max(len(f0), 1)

    if len(voiced_f0) < 3:
        return _empty_result("Insufficient voiced frames — likely silence or noise")

    pitch_mean = float(np.nanmean(voiced_f0))
    pitch_std = float(np.nanstd(voiced_f0))
    # Jitter = coefficient of variation (normalised by mean)
    pitch_jitter = pitch_std / pitch_mean if pitch_mean > 0 else 0.0

    # ------------------------------------------------------------------
    # 2. Hesitation / Pause Detection
    # ------------------------------------------------------------------
    pause_count, hesitation_ratio, longest_pause_ms = _detect_pauses(y, sr, duration_s)

    # ------------------------------------------------------------------
    # 3. Composite Stress Score (0–100)
    # ------------------------------------------------------------------
    # Weights derived from vocal analysis literature:
    #   - High jitter (>0.06) strongly indicates stress/nervousness
    #   - High hesitation ratio indicates uncertainty/deflection
    #   - Low voiced fraction can indicate a speaker trailing off
    jitter_component = min(pitch_jitter / 0.12, 1.0) * 50  # max 50 pts
    hesitation_component = min(hesitation_ratio / 0.30, 1.0) * 35  # max 35 pts
    voicing_component = max(0, 1.0 - voiced_fraction) * 15  # max 15 pts

    stress_score = jitter_component + hesitation_component + voicing_component
    stress_score = max(0.0, min(100.0, stress_score))

    # ------------------------------------------------------------------
    # 4. Human-readable interpretation
    # ------------------------------------------------------------------
    analysis = _interpret(stress_score, pitch_jitter, hesitation_ratio, pause_count, longest_pause_ms)

    return StressResult(
        stress_score=stress_score,
        pitch_mean_hz=pitch_mean,
        pitch_std_hz=pitch_std,
        pitch_jitter=pitch_jitter,
        hesitation_ratio=hesitation_ratio,
        voiced_fraction=voiced_fraction,
        pause_count=pause_count,
        longest_pause_ms=longest_pause_ms,
        analysis=analysis,
    )


def analyze_multiple_chunks(chunk_paths: list[str | Path]) -> StressResult:
    """Analyze several consecutive chunks and return an aggregated result.

    Useful for analysing a wider time window (e.g. the last 60 seconds).
    """
    results = []
    for p in chunk_paths:
        try:
            results.append(analyze_vocal_stress(p))
        except Exception as exc:
            _log(f"Skipping chunk {p}: {exc}")

    if not results:
        return _empty_result("No valid chunks to analyse")

    # Weighted average by voiced fraction (noisier chunks contribute less)
    weights = np.array([r.voiced_fraction for r in results], dtype=np.float64)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()

    agg = StressResult(
        stress_score=float(np.average([r.stress_score for r in results], weights=weights)),
        pitch_mean_hz=float(np.average([r.pitch_mean_hz for r in results], weights=weights)),
        pitch_std_hz=float(np.average([r.pitch_std_hz for r in results], weights=weights)),
        pitch_jitter=float(np.average([r.pitch_jitter for r in results], weights=weights)),
        hesitation_ratio=float(np.average([r.hesitation_ratio for r in results], weights=weights)),
        voiced_fraction=float(np.mean([r.voiced_fraction for r in results])),
        pause_count=sum(r.pause_count for r in results),
        longest_pause_ms=max(r.longest_pause_ms for r in results),
        analysis="",
    )
    agg.analysis = _interpret(
        agg.stress_score, agg.pitch_jitter, agg.hesitation_ratio,
        agg.pause_count, agg.longest_pause_ms,
    )
    return agg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_pauses(y: np.ndarray, sr: int, duration_s: float) -> tuple[int, float, float]:
    """Detect pauses >400ms using librosa.effects.split.

    Returns (pause_count, hesitation_ratio, longest_pause_ms).
    """
    # intervals of non-silent regions
    intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB, frame_length=2048, hop_length=512)

    if len(intervals) == 0:
        # Entire clip is silence
        return (1, 1.0, duration_s * 1000)

    threshold_samples = int(PAUSE_THRESHOLD_MS / 1000 * sr)
    total_pause_samples = 0
    pause_count = 0
    longest_pause_samples = 0

    # Gap before first voiced region
    if intervals[0][0] > 0:
        gap = intervals[0][0]
        if gap >= threshold_samples:
            total_pause_samples += gap
            pause_count += 1
            longest_pause_samples = max(longest_pause_samples, gap)

    # Gaps between voiced regions
    for i in range(1, len(intervals)):
        gap = intervals[i][0] - intervals[i - 1][1]
        if gap >= threshold_samples:
            total_pause_samples += gap
            pause_count += 1
            longest_pause_samples = max(longest_pause_samples, gap)

    # Gap after last voiced region
    trailing = len(y) - intervals[-1][1]
    if trailing >= threshold_samples:
        total_pause_samples += trailing
        pause_count += 1
        longest_pause_samples = max(longest_pause_samples, trailing)

    hesitation_ratio = total_pause_samples / max(len(y), 1)
    longest_pause_ms = longest_pause_samples / sr * 1000

    return pause_count, hesitation_ratio, longest_pause_ms


def _interpret(
    score: float, jitter: float, hesitation: float,
    pause_count: int, longest_pause_ms: float,
) -> str:
    """Generate a concise Wall Street-grade interpretation."""
    parts: list[str] = []

    if score >= 75:
        parts.append("HIGH STRESS DETECTED.")
    elif score >= 45:
        parts.append("ELEVATED STRESS.")
    elif score >= 20:
        parts.append("MODERATE — within normal range.")
    else:
        parts.append("LOW STRESS — speaker appears confident.")

    if jitter > 0.08:
        parts.append(f"Pitch jitter {jitter:.3f} indicates vocal tremor/nervousness.")
    elif jitter > 0.05:
        parts.append(f"Pitch jitter {jitter:.3f} is slightly elevated.")

    if hesitation > 0.20:
        parts.append(
            f"Hesitation ratio {hesitation:.1%} is high — "
            f"{pause_count} pauses detected (longest {longest_pause_ms:.0f}ms). "
            "Speaker may be choosing words carefully or deflecting."
        )
    elif hesitation > 0.10:
        parts.append(f"Moderate pausing ({pause_count} pauses, longest {longest_pause_ms:.0f}ms).")

    return " ".join(parts)


def _empty_result(reason: str) -> StressResult:
    return StressResult(
        stress_score=0.0,
        pitch_mean_hz=0.0,
        pitch_std_hz=0.0,
        pitch_jitter=0.0,
        hesitation_ratio=0.0,
        voiced_fraction=0.0,
        pause_count=0,
        longest_pause_ms=0.0,
        analysis=reason,
    )
