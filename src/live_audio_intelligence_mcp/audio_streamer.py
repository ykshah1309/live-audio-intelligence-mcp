"""
Audio Stream Ingestion Engine.

Extracts raw audio URLs from webcast links via yt-dlp and chunks live streams
into overlapping 15-second WAV buffers using ffmpeg subprocess for downstream
transcription and prosody analysis.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yt_dlp


def _log(msg: str) -> None:
    """All output goes to stderr — stdout is reserved for MCP JSON-RPC."""
    print(f"[AudioStreamer] {msg}", file=sys.stderr, flush=True)


@dataclass
class AudioChunk:
    """A single audio chunk with metadata."""

    path: Path
    stream_id: str
    chunk_index: int
    timestamp: float  # wall-clock time chunk was captured
    duration: float = 15.0  # seconds


@dataclass
class StreamState:
    """Mutable state for one active stream monitor."""

    stream_id: str
    url: str
    temp_dir: Path
    chunk_queue: queue.Queue[AudioChunk] = field(default_factory=lambda: queue.Queue(maxsize=200))
    chunk_index: int = 0
    stop_event: threading.Event = field(default_factory=threading.Event)
    ytdlp_process: subprocess.Popen | None = None
    ffmpeg_process: subprocess.Popen | None = None
    thread: threading.Thread | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None  # set when ffmpeg exits naturally
    vad_enabled: bool = True  # disable for low-SNR speakerphone audio


class StreamManager:
    """Manages live audio stream ingestion and chunking.

    Each monitored stream gets its own background thread, ffmpeg process,
    and temp directory. Chunks are 15-second overlapping WAV files written
    to a rolling queue that consumers (transcriber, analyzer) can drain.
    """

    CHUNK_DURATION = 15  # seconds
    OVERLAP = 3  # seconds of overlap between consecutive chunks
    SAMPLE_RATE = 16000  # 16kHz mono — optimal for Whisper

    def __init__(self) -> None:
        self._streams: dict[str, StreamState] = {}
        self._lock = threading.Lock()

    @property
    def active_streams(self) -> list[str]:
        with self._lock:
            return list(self._streams.keys())

    def get_state(self, stream_id: str) -> StreamState | None:
        with self._lock:
            return self._streams.get(stream_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_stream(self, url: str, *, vad_enabled: bool = True) -> str:
        """Extract audio URL and start the chunking background thread.

        Args:
            url: Webcast URL to ingest.
            vad_enabled: If False, disable Whisper's VAD filter. Useful for
                low-SNR audio (speakerphone calls) where Silero VAD
                aggressively drops speech as silence.

        Returns a unique stream_id for subsequent operations.
        """
        stream_id = uuid.uuid4().hex[:12]
        # Quick validation pass — fail fast if yt-dlp can't resolve the URL.
        # We don't use the resolved URL directly because for DASH / fragmented
        # streams (YouTube, Twitch, live TV) ffmpeg can't follow the format
        # without yt-dlp's fragment muxer. Instead we pipe yt-dlp's stdout
        # into ffmpeg in _chunking_loop.
        _log(f"Validating URL: {url}")
        self._validate_url(url)

        temp_dir = Path(tempfile.mkdtemp(prefix=f"lai_{stream_id}_"))
        state = StreamState(
            stream_id=stream_id,
            url=url,
            temp_dir=temp_dir,
            vad_enabled=vad_enabled,
        )

        worker = threading.Thread(
            target=self._chunking_loop,
            args=(state,),
            name=f"stream-{stream_id}",
            daemon=True,
        )
        state.thread = worker

        with self._lock:
            self._streams[stream_id] = state

        worker.start()
        _log(f"Stream {stream_id} ingestion started")
        return stream_id

    def stop_stream(self, stream_id: str) -> bool:
        """Signal the background thread to stop, kill ffmpeg, clean up."""
        with self._lock:
            state = self._streams.pop(stream_id, None)

        if state is None:
            return False

        state.stop_event.set()
        self._kill_ffmpeg(state)

        if state.thread and state.thread.is_alive():
            state.thread.join(timeout=5)

        # Drain remaining chunks and remove temp files
        self._cleanup_temp(state)
        _log(f"Stream {stream_id} stopped and cleaned up")
        return True

    def stop_all(self) -> None:
        """Stop every active stream — used during shutdown."""
        with self._lock:
            ids = list(self._streams.keys())
        for sid in ids:
            self.stop_stream(sid)

    def get_chunks(self, stream_id: str, max_chunks: int = 0) -> list[AudioChunk]:
        """Drain up to *max_chunks* from the queue (0 = all available)."""
        state = self.get_state(stream_id)
        if state is None:
            return []

        chunks: list[AudioChunk] = []
        while True:
            if 0 < max_chunks <= len(chunks):
                break
            try:
                chunks.append(state.chunk_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_url(url: str) -> None:
        """Fail fast with a clear error if yt-dlp can't resolve the URL.

        We don't need the resolved format URL — _chunking_loop pipes yt-dlp's
        stdout into ffmpeg. This is a pre-flight check so the caller hears
        "404 / unsupported site" immediately instead of waiting for the
        chunking thread to silently die.
        """
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio/best",
            "skip_download": True,
            "logger": type("L", (), {
                "debug": lambda self, msg: None,
                "warning": lambda self, msg: _log(f"yt-dlp warn: {msg}"),
                "error": lambda self, msg: _log(f"yt-dlp err: {msg}"),
            })(),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise RuntimeError(f"yt-dlp returned no info for: {url}")

    def _chunking_loop(self, state: StreamState) -> None:
        """Background thread: runs yt-dlp → ffmpeg and slices output into chunks.

        yt-dlp downloads / mux-demuxes the webcast, feeding bytes to ffmpeg
        which transcodes to 16kHz mono PCM. We read fixed-size blocks from
        ffmpeg's stdout, write them as WAV files into the temp dir, and push
        AudioChunk refs onto the queue.

        When the webcast ends naturally (or the network drops), the try/finally
        marks the stream as ended but keeps its StreamState in _streams so the
        transcription loop can finish draining the queue and analyze_speaker_stress
        can still read chunk files afterwards. Cleanup of temp files happens
        only when stop_monitor is called (or the server shuts down).
        """
        import wave

        segment_samples = self.CHUNK_DURATION * self.SAMPLE_RATE
        overlap_samples = self.OVERLAP * self.SAMPLE_RATE
        bytes_per_sample = 2  # 16-bit PCM
        segment_bytes = segment_samples * bytes_per_sample
        overlap_bytes = overlap_samples * bytes_per_sample

        # Pipeline: yt-dlp downloads/muxes the stream to stdout, ffmpeg
        # transcodes it to 16kHz s16le PCM. yt-dlp handles every URL type
        # (DASH fragments, HLS, progressive MP3, Twitch, YouTube live) so
        # ffmpeg doesn't have to.
        ytdlp_cmd = [
            sys.executable, "-m", "yt_dlp",
            "-q", "--no-warnings",
            "-f", "bestaudio/best",
            "-o", "-",  # stdout
            "--no-part",
            state.url,
        ]
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ]

        try:
            # stdin=DEVNULL is critical: when the MCP server is launched over
            # stdio, the server's stdin is the JSON-RPC pipe from the client.
            # Without this, yt-dlp inherits that stdin and may block reading
            # protocol bytes (or worse, consume messages intended for us).
            state.ytdlp_process = subprocess.Popen(
                ytdlp_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            _log("ERROR: python / yt-dlp not found on PATH")
            self._reap_dead_stream(state)
            return
        except Exception as exc:
            _log(f"ERROR starting yt-dlp: {exc}")
            self._reap_dead_stream(state)
            return

        try:
            state.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=state.ytdlp_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            # Allow yt-dlp to receive SIGPIPE if ffmpeg exits first.
            if state.ytdlp_process.stdout is not None:
                state.ytdlp_process.stdout.close()
        except FileNotFoundError:
            _log("ERROR: ffmpeg not found on PATH")
            self._kill_subprocess(state.ytdlp_process)
            self._reap_dead_stream(state)
            return
        except Exception as exc:
            _log(f"ERROR starting ffmpeg: {exc}")
            self._kill_subprocess(state.ytdlp_process)
            self._reap_dead_stream(state)
            return

        # Drain stderr from both subprocesses on daemon threads.
        threading.Thread(
            target=self._drain_stderr,
            args=(state.ytdlp_process, f"yt-dlp[{state.stream_id}]"),
            name=f"ytdlp-stderr-{state.stream_id}",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._drain_stderr,
            args=(state.ffmpeg_process, f"ffmpeg[{state.stream_id}]"),
            name=f"ffmpeg-stderr-{state.stream_id}",
            daemon=True,
        ).start()

        _log(
            f"yt-dlp PID {state.ytdlp_process.pid} -> "
            f"ffmpeg PID {state.ffmpeg_process.pid} capturing stream {state.stream_id}"
        )

        try:
            carryover = b""
            while not state.stop_event.is_set():
                proc = state.ffmpeg_process
                if proc is None or proc.stdout is None:
                    break

                needed = segment_bytes - len(carryover)
                raw = proc.stdout.read(needed)
                if not raw:
                    # Stream ended naturally (webcast over, network drop).
                    # Flush any remaining carryover as a final (short) chunk
                    # so short streams still produce usable output.
                    if len(carryover) >= self.SAMPLE_RATE * bytes_per_sample:
                        final_path = state.temp_dir / f"chunk_{state.chunk_index:06d}.wav"
                        try:
                            with wave.open(str(final_path), "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(bytes_per_sample)
                                wf.setframerate(self.SAMPLE_RATE)
                                wf.writeframes(carryover)
                            final_chunk = AudioChunk(
                                path=final_path,
                                stream_id=state.stream_id,
                                chunk_index=state.chunk_index,
                                timestamp=time.time(),
                                duration=len(carryover) / (self.SAMPLE_RATE * bytes_per_sample),
                            )
                            try:
                                state.chunk_queue.put(final_chunk, timeout=2)
                                state.chunk_index += 1
                                _log(
                                    f"Stream {state.stream_id}: flushed final "
                                    f"partial chunk ({final_chunk.duration:.1f}s)"
                                )
                            except queue.Full:
                                pass
                        except Exception as exc:
                            _log(f"Error writing final partial chunk: {exc}")
                    _log(f"Stream {state.stream_id}: ffmpeg stdout closed (stream ended)")
                    break

                pcm_data = carryover + raw
                if len(pcm_data) < segment_bytes:
                    carryover = pcm_data
                    continue

                # Write WAV chunk
                chunk_path = state.temp_dir / f"chunk_{state.chunk_index:06d}.wav"
                try:
                    with wave.open(str(chunk_path), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(bytes_per_sample)
                        wf.setframerate(self.SAMPLE_RATE)
                        wf.writeframes(pcm_data[:segment_bytes])
                except Exception as exc:
                    _log(f"Error writing chunk: {exc}")
                    carryover = b""
                    continue

                chunk = AudioChunk(
                    path=chunk_path,
                    stream_id=state.stream_id,
                    chunk_index=state.chunk_index,
                    timestamp=time.time(),
                    duration=self.CHUNK_DURATION,
                )

                try:
                    state.chunk_queue.put(chunk, timeout=2)
                except queue.Full:
                    # Drop oldest to prevent backpressure from stalling ingestion
                    try:
                        dropped = state.chunk_queue.get_nowait()
                        self._safe_delete(dropped.path)
                    except queue.Empty:
                        pass
                    state.chunk_queue.put(chunk, timeout=2)

                state.chunk_index += 1

                # Keep the overlap tail for continuity
                carryover = pcm_data[segment_bytes - overlap_bytes:]
        except Exception as exc:
            _log(f"Unexpected error in chunking loop for {state.stream_id}: {exc}")
        finally:
            # --- Critical cleanup: prevent phantom streams ---
            # If we reach here and stop_event was NOT set, ffmpeg died on its
            # own (webcast ended, crash, network drop). We must reap the state
            # so it doesn't sit in _streams leaking memory forever.
            if not state.stop_event.is_set():
                _log(f"Stream {state.stream_id}: auto-reaping after natural exit")
                self._reap_dead_stream(state)
            _log(f"Chunking loop exited for stream {state.stream_id}")

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen | None, tag: str) -> None:
        """Forward a subprocess's stderr into our log so failures are visible.

        Runs on a daemon thread until stderr closes. Without this, a full
        stderr pipe would eventually block the subprocess itself.
        """
        if proc is None or proc.stderr is None:
            return
        try:
            for raw_line in proc.stderr:
                try:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    continue
                if line:
                    _log(f"{tag}: {line}")
        except Exception:
            pass

    @staticmethod
    def _kill_subprocess(proc: subprocess.Popen | None) -> None:
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except Exception:
            pass

    def _kill_ffmpeg(self, state: StreamState) -> None:
        self._kill_subprocess(state.ffmpeg_process)
        state.ffmpeg_process = None
        self._kill_subprocess(state.ytdlp_process)
        state.ytdlp_process = None

    def _reap_dead_stream(self, state: StreamState) -> None:
        """Mark a stream whose ffmpeg process died unexpectedly as ended.

        Called from the chunking thread when ffmpeg exits on its own
        (webcast ended, network drop, crash). We KEEP the state in _streams
        and preserve its temp_dir so:
          * the transcription loop can drain any queued/flushed chunks
          * analyze_speaker_stress can still read chunk files after the
            stream ends

        The state is cleaned up only when the user calls stop_monitor
        or the server shuts down via stop_all().
        """
        state.ended_at = time.time()
        self._kill_ffmpeg(state)
        _log(f"Stream {state.stream_id} marked ended (ffmpeg exit)")

    @staticmethod
    def _cleanup_temp(state: StreamState) -> None:
        try:
            shutil.rmtree(state.temp_dir, ignore_errors=True)
        except Exception:
            pass

    @staticmethod
    def _safe_delete(path: Path) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass
