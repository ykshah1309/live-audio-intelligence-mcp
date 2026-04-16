"""End-to-end smoke test for live-audio-intelligence-mcp.

Spawns the MCP server as a subprocess over stdio, lists its tools, generates
a synthetic 30s audio WAV (speech-like tone + silence gaps), writes it to a
temp dir as if the StreamManager had produced it, and directly exercises the
prosody analyzer + transcriber on it to validate the pipeline end to end.

Run:
    python scripts/smoke_test.py
(activate your venv first so `python` is the one with the project installed)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def gen_synthetic_speech_wav(path: Path, duration_s: float = 16.0, sr: int = 16000) -> None:
    """Create a synthetic speech-like WAV — frequency-modulated tone with pauses."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    # Base F0 varying 120-220 Hz (human voice-like)
    f0 = 150 + 30 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = 0.3 * np.sin(phase)
    # Harmonics — crude formants
    signal += 0.15 * np.sin(2 * phase)
    signal += 0.08 * np.sin(3 * phase)
    # Insert 3 pauses of 600ms each
    for start_s in (3.0, 7.5, 12.0):
        a = int(start_s * sr)
        b = a + int(0.6 * sr)
        signal[a:b] = 0.0
    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def test_prosody_direct() -> None:
    from live_audio_intelligence_mcp.prosody_analyzer import (
        analyze_multiple_chunks,
        analyze_vocal_stress,
    )
    print("\n[1] Prosody analyzer — direct test")
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "chunk_000001.wav"
        p2 = Path(td) / "chunk_000002.wav"
        gen_synthetic_speech_wav(p1, duration_s=15.0)
        gen_synthetic_speech_wav(p2, duration_s=15.0)

        r = analyze_vocal_stress(p1)
        print(f"  single chunk stress_score: {r.stress_score:.1f}")
        print(f"  pitch_mean_hz: {r.pitch_mean_hz:.1f}")
        print(f"  pitch_jitter:  {r.pitch_jitter:.4f}")
        print(f"  hesitation:    {r.hesitation_ratio:.3f}")
        print(f"  pauses:        {r.pause_count}")
        print(f"  analysis:      {r.analysis}")
        assert 0 <= r.stress_score <= 100, "stress_score out of range"
        assert r.pitch_mean_hz > 0, "pitch_mean_hz should be positive for synthetic tone"
        assert r.pause_count >= 1, "should detect the injected pauses"

        agg = analyze_multiple_chunks([p1, p2])
        print(f"  aggregate stress_score: {agg.stress_score:.1f} (2 chunks)")
        assert 0 <= agg.stress_score <= 100


def test_stream_manager_direct() -> None:
    """Validate StreamManager internals that we can exercise without ffmpeg-ingest."""
    from live_audio_intelligence_mcp.audio_streamer import StreamManager
    print("\n[2] StreamManager — lifecycle test (no live URL)")
    sm = StreamManager()
    # get_state for unknown id returns None
    assert sm.get_state("nope") is None
    assert sm.stop_stream("nope") is False
    # stop_all on empty manager should not raise
    sm.stop_all()
    print("  lifecycle boundaries OK")


def test_transcriber_direct() -> None:
    from live_audio_intelligence_mcp.audio_streamer import AudioChunk
    from live_audio_intelligence_mcp.transcriber import Transcriber
    print("\n[3] Transcriber — direct test (loads Whisper base.en)")
    t0 = time.time()
    t = Transcriber(model_size="base.en", compute_type="int8", device="cpu")
    print(f"  model load: {time.time() - t0:.1f}s")

    with tempfile.TemporaryDirectory() as td:
        wp = Path(td) / "chunk_000000.wav"
        gen_synthetic_speech_wav(wp, duration_s=5.0)
        chunk = AudioChunk(path=wp, stream_id="test", chunk_index=0, timestamp=time.time())
        segs = t.transcribe_chunk(chunk, vad_enabled=False)
        print(f"  segments produced: {len(segs)}")
        # Synthetic tone won't produce real words but pipeline should not crash
        text = t.get_rolling_text("test")
        print(f"  rolling_text: {text[:80]!r}")
    print("  transcriber OK")


async def test_mcp_protocol() -> None:
    """Spawn server as subprocess, perform MCP handshake, list tools."""
    print("\n[4] MCP protocol — subprocess handshake")
    server_cmd = [sys.executable, "-m", "live_audio_intelligence_mcp"]
    env = dict(os.environ)
    proc = await asyncio.create_subprocess_exec(
        *server_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    async def send(req: dict) -> None:
        payload = (json.dumps(req) + "\n").encode()
        proc.stdin.write(payload)
        await proc.stdin.drain()

    async def recv_line(timeout: float = 60.0) -> dict:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
        return json.loads(line.decode())

    try:
        # 1. initialize
        await send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "0.1"},
            },
        })
        init = await recv_line()
        print(f"  initialize -> server: {init.get('result', {}).get('serverInfo', {})}")
        assert "result" in init

        # 2. initialized notification
        await send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 3. tools/list
        await send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools_resp = await recv_line()
        tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
        print(f"  tools/list -> {tool_names}")
        expected = {"monitor_live_stream", "get_rolling_transcript",
                    "analyze_speaker_stress", "stop_monitor"}
        assert set(tool_names) == expected, f"tool mismatch: {tool_names}"

        # Validate schema of monitor_live_stream
        ml = next(t for t in tools_resp["result"]["tools"] if t["name"] == "monitor_live_stream")
        props = ml["inputSchema"]["properties"]
        assert "url" in props, "url param missing"
        assert "disable_vad" in props, "disable_vad param missing"
        print(f"  monitor_live_stream params: {list(props.keys())}")
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    print("  MCP protocol OK")


def main() -> None:
    test_prosody_direct()
    test_stream_manager_direct()
    test_transcriber_direct()
    asyncio.run(test_mcp_protocol())
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
