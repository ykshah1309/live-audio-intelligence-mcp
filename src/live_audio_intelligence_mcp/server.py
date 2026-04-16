"""
MCP Server Interface — the FastMCP wrapper.

Exposes four tools over JSON-RPC stdio transport:
  1. monitor_live_stream  — start ingesting a webcast
  2. get_rolling_transcript — retrieve recent transcript text
  3. analyze_speaker_stress — run prosody analysis on recent audio
  4. stop_monitor          — tear down a stream

All heavy/blocking work (Whisper inference, FFmpeg I/O, librosa DSP) is
dispatched to threads via asyncio.to_thread() so the MCP event loop stays
responsive.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field

from mcp.server.fastmcp import Context, FastMCP

from .audio_streamer import StreamManager
from .exceptions import UnknownStreamError
from .prosody_analyzer import StressResult, analyze_multiple_chunks
from .transcriber import Transcriber


def _log(msg: str) -> None:
    print(f"[MCPServer] {msg}", file=sys.stderr, flush=True)


def _warmup_prosody() -> None:
    """Force librosa.pyin and librosa.effects.split to JIT-compile at startup.

    See app_lifespan for rationale. Runs a 1-second synthetic tone through the
    same code paths analyze_vocal_stress uses so all numba kernels get cached.
    """
    import numpy as _np
    import librosa as _librosa
    t = _np.linspace(0, 1.0, 16000, endpoint=False)
    y = (0.3 * _np.sin(2 * _np.pi * 220 * t)).astype(_np.float32)
    _log("Warming up librosa.pyin JIT …")
    _librosa.pyin(y, sr=16000, fmin=65.0, fmax=600.0, frame_length=2048)
    _librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    _log("librosa.pyin warm-up complete")


# ======================================================================
# Lifespan — heavy resources initialised once at startup
# ======================================================================

@dataclass
class AppContext:
    """Shared application state available to every tool invocation."""

    stream_manager: StreamManager
    transcriber: Transcriber


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Startup: load Whisper model + create StreamManager.
    Shutdown: stop all streams and free resources.
    """
    _log("Initialising LiveAudioIntelligence resources…")
    stream_manager = StreamManager()

    # Load the model in a thread so we don't block the loop
    transcriber = await asyncio.to_thread(
        Transcriber,
        model_size="base.en",
        compute_type="int8",
        device="cpu",
    )

    # Warm up librosa.pyin's numba-JIT code on a short synthetic signal.
    # The first invocation of pyin triggers numba compilation (llvmlite IR
    # generation + machine-code lowering). If that first call happens on a
    # worker thread spawned via asyncio.to_thread *after* MCP stdio has
    # taken over, it can hang on Windows. Warming up during lifespan avoids
    # that entirely — the JIT caches are already populated by the time any
    # tool call arrives.
    await asyncio.to_thread(_warmup_prosody)

    _log("All resources ready.")
    try:
        yield AppContext(stream_manager=stream_manager, transcriber=transcriber)
    finally:
        _log("Shutting down — stopping all streams…")
        stream_manager.stop_all()
        _log("Shutdown complete.")


# ======================================================================
# FastMCP server instance
# ======================================================================

mcp = FastMCP(
    "LiveAudioIntelligence",
    instructions=(
        "Institutional-grade live audio intelligence server for Wall Street. "
        "Ingests financial webcasts, transcribes in near real-time with "
        "faster-whisper, and analyses vocal prosody to detect executive stress "
        "and hesitation patterns that precede market-moving events."
    ),
    lifespan=app_lifespan,
)


# ======================================================================
# Pydantic schemas for structured output
# ======================================================================

class StreamInfo(BaseModel):
    stream_id: str = Field(description="Unique identifier for the monitored stream")
    url: str = Field(description="Original URL being monitored")
    status: str = Field(description="Current stream status")


class TranscriptResult(BaseModel):
    stream_id: str
    minutes_back: int
    text: str = Field(description="Concatenated transcript text")
    segment_count: int = Field(description="Number of transcript segments in window")


class StressAnalysis(BaseModel):
    stream_id: str
    time_window_seconds: int
    stress_score: float = Field(ge=0, le=100, description="Composite vocal stress score 0-100")
    pitch_mean_hz: float = Field(description="Mean fundamental frequency in Hz")
    pitch_std_hz: float = Field(description="Standard deviation of F0")
    pitch_jitter: float = Field(description="Normalised pitch jitter (coefficient of variation)")
    hesitation_ratio: float = Field(ge=0, le=1, description="Fraction of audio that is silence >400ms")
    voiced_fraction: float = Field(ge=0, le=1, description="Fraction of voiced frames")
    pause_count: int = Field(description="Number of significant pauses detected")
    longest_pause_ms: float = Field(description="Duration of longest pause in ms")
    analysis: str = Field(description="Human-readable stress interpretation")
    chunks_analyzed: int = Field(description="Number of audio chunks processed")


class StopResult(BaseModel):
    stream_id: str
    status: str
    duration_seconds: float = Field(description="How long the stream was monitored")


# ======================================================================
# Tool: monitor_live_stream
# ======================================================================

@mcp.tool()
async def monitor_live_stream(
    url: str,
    ctx: Context,
    disable_vad: Annotated[
        bool,
        Field(
            description=(
                "Disable Silero VAD filtering. Set to true for low-quality "
                "speakerphone audio where VAD aggressively drops speech as silence."
            ),
        ),
    ] = False,
) -> StreamInfo:
    """Start monitoring a live financial webcast for transcription and stress analysis.

    Provide a URL to a live earnings call, CNBC stream, or any webcast.
    The server will extract the audio stream, begin chunking it into
    15-second segments, and continuously transcribe them in the background.

    If the audio is low-quality (speakerphone, poor connection), set
    disable_vad=true to prevent the voice activity detector from
    aggressively dropping muddy speech segments.

    Returns a stream_id you'll use for all subsequent operations.
    """
    app: AppContext = ctx.request_context.lifespan_context
    await ctx.info(f"Extracting audio stream from: {url}")

    vad_enabled = not disable_vad

    # yt-dlp + ffmpeg startup are blocking — run in thread
    try:
        stream_id = await asyncio.to_thread(
            app.stream_manager.start_stream, url, vad_enabled=vad_enabled,
        )
    except Exception as exc:
        await ctx.error(f"Failed to start stream: {exc}")
        raise

    # Create the transcript buffer
    app.transcriber.get_or_create_transcript(stream_id)

    # Kick off a background task to continuously transcribe new chunks
    asyncio.create_task(_transcription_loop(app, stream_id))

    vad_note = " (VAD disabled — low-SNR mode)" if disable_vad else ""
    await ctx.info(f"Stream {stream_id} is live — transcription running{vad_note}")
    return StreamInfo(stream_id=stream_id, url=url, status="monitoring")


async def _transcription_loop(app: AppContext, stream_id: str) -> None:
    """Background coroutine that drains audio chunks and transcribes them.

    Runs until the stream is stopped OR the stream has ended AND its queue
    is drained. Transcription is offloaded to a thread so the event loop
    stays free.
    """
    _log(f"Transcription loop started for {stream_id}")
    while True:
        state = app.stream_manager.get_state(stream_id)
        if state is None:
            _log(f"Stream {stream_id} gone — transcription loop exiting")
            break

        vad_enabled = state.vad_enabled
        chunks = app.stream_manager.get_chunks(stream_id, max_chunks=5)
        if chunks:
            try:
                await asyncio.to_thread(
                    app.transcriber.transcribe_chunks, chunks,
                    vad_enabled=vad_enabled,
                )
            except Exception as exc:
                _log(f"Transcription error: {exc}")
        else:
            # Queue is empty — if the stream has also ended, we're done.
            if state.ended_at is not None:
                _log(f"Stream {stream_id} ended and queue drained — transcription loop exiting")
                break
            await asyncio.sleep(1.0)


# ======================================================================
# Tool: get_rolling_transcript
# ======================================================================

@mcp.tool()
async def get_rolling_transcript(
    stream_id: str,
    minutes_back: Annotated[int, Field(ge=1, le=30, description="How many minutes of transcript to retrieve")] = 10,
    ctx: Context = None,  # type: ignore[assignment]
) -> TranscriptResult:
    """Retrieve the rolling transcript from a monitored stream.

    Returns the concatenated text from the last N minutes, ideal for
    feeding into an LLM for summarisation or sentiment analysis of the
    earnings call in progress.
    """
    app: AppContext = ctx.request_context.lifespan_context
    transcript = app.transcriber.get_or_create_transcript(stream_id)
    text = transcript.get_text(minutes_back)
    segments = transcript.get_segments(minutes_back * 60)

    return TranscriptResult(
        stream_id=stream_id,
        minutes_back=minutes_back,
        text=text if text else "(no transcript available yet — stream may still be buffering)",
        segment_count=len(segments),
    )


# ======================================================================
# Tool: analyze_speaker_stress
# ======================================================================

@mcp.tool()
async def analyze_speaker_stress(
    stream_id: str,
    time_window_seconds: Annotated[int, Field(ge=15, le=300, description="Analysis window in seconds")] = 60,
    ctx: Context = None,  # type: ignore[assignment]
) -> StressAnalysis:
    """Analyse the speaker's vocal stress over a recent time window.

    Extracts F0 pitch contour, measures pitch jitter (vocal tremor),
    and detects hesitation patterns (pauses > 400ms). Returns a
    composite stress score from 0–100 where:

      0–20  = confident, fluent delivery
      20–45 = normal variation
      45–75 = elevated stress — worth monitoring
      75–100 = high stress — potential market-moving signal

    Higher scores correlate with executive nervousness, evasion, and
    uncertainty — the kind of prosodic signals that precede guidance
    revisions and earnings misses.
    """
    app: AppContext = ctx.request_context.lifespan_context

    state = app.stream_manager.get_state(stream_id)
    if state is None:
        raise UnknownStreamError(f"No active stream with id: {stream_id}")

    # Collect chunk files that fall within the time window
    cutoff = time.time() - time_window_seconds
    chunk_files = sorted(state.temp_dir.glob("chunk_*.wav"))
    recent_files = []
    for cf in chunk_files:
        try:
            if os.path.getmtime(cf) >= cutoff:
                recent_files.append(cf)
        except OSError:
            continue

    if not recent_files:
        raise ValueError(
            f"No audio chunks available in the last {time_window_seconds}s. "
            "The stream may still be buffering."
        )

    await ctx.report_progress(progress=0.3, total=1.0, message=f"Analysing {len(recent_files)} chunks…")

    # Run the CPU-heavy analysis off the event loop
    result: StressResult = await asyncio.to_thread(
        analyze_multiple_chunks, recent_files,
    )

    await ctx.report_progress(progress=1.0, total=1.0, message="Analysis complete")

    return StressAnalysis(
        stream_id=stream_id,
        time_window_seconds=time_window_seconds,
        stress_score=round(result.stress_score, 1),
        pitch_mean_hz=round(result.pitch_mean_hz, 1),
        pitch_std_hz=round(result.pitch_std_hz, 2),
        pitch_jitter=round(result.pitch_jitter, 4),
        hesitation_ratio=round(result.hesitation_ratio, 4),
        voiced_fraction=round(result.voiced_fraction, 4),
        pause_count=result.pause_count,
        longest_pause_ms=round(result.longest_pause_ms, 1),
        analysis=result.analysis,
        chunks_analyzed=len(recent_files),
    )


# ======================================================================
# Tool: stop_monitor
# ======================================================================

@mcp.tool()
async def stop_monitor(
    stream_id: str,
    ctx: Context = None,  # type: ignore[assignment]
) -> StopResult:
    """Stop monitoring a live stream and clean up all resources.

    Kills the ffmpeg process, removes temporary audio files, and
    clears the transcript buffer.
    """
    app: AppContext = ctx.request_context.lifespan_context

    state = app.stream_manager.get_state(stream_id)
    duration = time.time() - state.started_at if state else 0

    stopped = await asyncio.to_thread(app.stream_manager.stop_stream, stream_id)
    app.transcriber.remove_transcript(stream_id)

    if not stopped:
        raise UnknownStreamError(f"No active stream with id: {stream_id}")

    await ctx.info(f"Stream {stream_id} stopped after {duration:.0f}s")
    return StopResult(
        stream_id=stream_id,
        status="stopped",
        duration_seconds=round(duration, 1),
    )


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    """CLI entry point — runs the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
