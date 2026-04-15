"""
Whisper Transcription Pipeline.

Wraps faster-whisper with a rolling transcript buffer that retains the last
N minutes of speech, complete with timestamps. Designed to be fed audio chunks
from the StreamManager's queue and run inference off the main event loop.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from faster_whisper import WhisperModel

from .audio_streamer import AudioChunk


def _log(msg: str) -> None:
    print(f"[Transcriber] {msg}", file=sys.stderr, flush=True)


@dataclass
class TranscriptSegment:
    """One segment of transcribed speech."""

    text: str
    start: float  # seconds relative to chunk start
    end: float  # seconds relative to chunk start
    chunk_index: int
    wall_time: float  # wall-clock time the chunk was captured
    confidence: float = 0.0


@dataclass
class StreamTranscript:
    """Rolling transcript buffer for a single stream."""

    stream_id: str
    max_age_seconds: float = 600.0  # 10 minutes default
    segments: deque[TranscriptSegment] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, segment: TranscriptSegment) -> None:
        with self._lock:
            self.segments.append(segment)
            self._prune()

    def get_text(self, minutes_back: int = 10) -> str:
        """Return concatenated transcript text for the last N minutes."""
        cutoff = time.time() - (minutes_back * 60)
        with self._lock:
            lines = [
                seg.text
                for seg in self.segments
                if seg.wall_time >= cutoff
            ]
        return " ".join(lines).strip()

    def get_segments(self, seconds_back: int = 60) -> list[TranscriptSegment]:
        """Return raw segments for the last N seconds."""
        cutoff = time.time() - seconds_back
        with self._lock:
            return [
                seg for seg in self.segments
                if seg.wall_time >= cutoff
            ]

    def _prune(self) -> None:
        """Remove segments older than max_age_seconds."""
        cutoff = time.time() - self.max_age_seconds
        while self.segments and self.segments[0].wall_time < cutoff:
            self.segments.popleft()


class Transcriber:
    """Manages the faster-whisper model and per-stream transcript buffers.

    The model is loaded once on init (typically during MCP server lifespan
    startup) and shared across all streams. Inference is performed in
    blocking calls that the server wraps with asyncio.to_thread().
    """

    def __init__(
        self,
        model_size: str = "base.en",
        compute_type: str = "int8",
        device: str = "cpu",
    ) -> None:
        _log(f"Loading Whisper model: {model_size} ({compute_type}) on {device}")
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
        )
        _log("Whisper model loaded")

        self._transcripts: dict[str, StreamTranscript] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Transcript buffer management
    # ------------------------------------------------------------------

    def get_or_create_transcript(self, stream_id: str) -> StreamTranscript:
        with self._lock:
            if stream_id not in self._transcripts:
                self._transcripts[stream_id] = StreamTranscript(stream_id=stream_id)
            return self._transcripts[stream_id]

    def remove_transcript(self, stream_id: str) -> None:
        with self._lock:
            self._transcripts.pop(stream_id, None)

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    def transcribe_chunk(
        self, chunk: AudioChunk, *, vad_enabled: bool = True,
    ) -> list[TranscriptSegment]:
        """Transcribe a single audio chunk and store results.

        Args:
            chunk: The audio chunk to transcribe.
            vad_enabled: If False, skip Silero VAD filtering. Set this to
                False for low-SNR audio (e.g. speakerphone earnings calls)
                where VAD aggressively classifies muddy speech as silence.

        This is a blocking call — should be run via asyncio.to_thread().
        Returns the new segments produced.
        """
        path = chunk.path
        if not path.exists():
            _log(f"Chunk file missing: {path}")
            return []

        try:
            segments_gen, info = self._model.transcribe(
                str(path),
                beam_size=3,  # balance speed vs accuracy
                language="en",
                vad_filter=vad_enabled,
                vad_parameters=dict(min_silence_duration_ms=300) if vad_enabled else None,
                word_timestamps=False,
                condition_on_previous_text=False,
            )

            # Materialize the generator (inference happens here)
            raw_segments = list(segments_gen)
        except Exception as exc:
            _log(f"Transcription error on {path.name}: {exc}")
            return []

        transcript = self.get_or_create_transcript(chunk.stream_id)
        new_segments: list[TranscriptSegment] = []

        for seg in raw_segments:
            ts = TranscriptSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                chunk_index=chunk.chunk_index,
                wall_time=chunk.timestamp + seg.start,
                confidence=seg.avg_logprob,
            )
            if ts.text:
                transcript.append(ts)
                new_segments.append(ts)

        return new_segments

    def transcribe_chunks(
        self, chunks: list[AudioChunk], *, vad_enabled: bool = True,
    ) -> list[TranscriptSegment]:
        """Transcribe a batch of chunks sequentially. Blocking call."""
        all_segments: list[TranscriptSegment] = []
        for chunk in chunks:
            all_segments.extend(self.transcribe_chunk(chunk, vad_enabled=vad_enabled))
        return all_segments

    def get_rolling_text(self, stream_id: str, minutes_back: int = 10) -> str:
        """Return the rolling transcript text for a stream."""
        transcript = self.get_or_create_transcript(stream_id)
        return transcript.get_text(minutes_back)
