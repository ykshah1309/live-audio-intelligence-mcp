"""Full live-URL test — drives monitor_live_stream through the MCP server
against a real short YouTube video, waits for chunks, then calls
get_rolling_transcript and analyze_speaker_stress.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Public-domain MP3 hosted on SampleLib (~15 sec). More representative of a
# typical webcast audio stream than YouTube, and avoids YouTube's JS-challenge
# machinery that intermittently fails without a JS runtime installed.
LIVE_URL = "https://download.samplelib.com/mp3/sample-15s.mp3"


async def main() -> None:
    server_cmd = [sys.executable, "-m", "live_audio_intelligence_mcp"]
    env = dict(os.environ)
    proc = await asyncio.create_subprocess_exec(
        *server_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
    )

    _next_id = 0

    async def send(method: str, params: dict | None = None, notify: bool = False) -> dict | None:
        nonlocal _next_id
        req: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            req["params"] = params
        if not notify:
            _next_id += 1
            req["id"] = _next_id
        proc.stdin.write((json.dumps(req) + "\n").encode())
        await proc.stdin.drain()
        if notify:
            return None
        # Skip notifications from server until we see a response
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=120)
            if not line:
                raise RuntimeError("server exited")
            msg = json.loads(line.decode())
            if "id" in msg:
                return msg

    try:
        await send("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "live-test", "version": "0.1"},
        })
        await send("notifications/initialized", notify=True)

        print(f"[1] call monitor_live_stream on {LIVE_URL}")
        resp = await send("tools/call", {
            "name": "monitor_live_stream",
            "arguments": {"url": LIVE_URL, "disable_vad": False},
        })
        content = resp["result"]["content"]
        print(f"    result: {content}")
        # Structured content holds the stream_id
        struct = resp["result"].get("structuredContent", {})
        stream_id = struct.get("stream_id") or json.loads(content[0]["text"]).get("stream_id")
        print(f"    stream_id = {stream_id}")
        assert stream_id

        # Wait ~20s for at least one chunk to be produced
        print("[2] waiting 25s for chunks…")
        await asyncio.sleep(25)

        print("[3] call get_rolling_transcript")
        resp = await send("tools/call", {
            "name": "get_rolling_transcript",
            "arguments": {"stream_id": stream_id, "minutes_back": 5},
        })
        tr = resp["result"]
        print(f"    isError:    {tr.get('isError')}")
        print(f"    structured: {tr.get('structuredContent')}")

        print("[4] call analyze_speaker_stress")
        resp = await send("tools/call", {
            "name": "analyze_speaker_stress",
            "arguments": {"stream_id": stream_id, "time_window_seconds": 30},
        })
        sa = resp.get("result") or resp.get("error")
        print(f"    isError:    {sa.get('isError') if isinstance(sa, dict) else None}")
        print(f"    content:    {sa.get('content') if isinstance(sa, dict) else sa}")
        print(f"    structured: {sa.get('structuredContent') if isinstance(sa, dict) else None}")

        print("[5] call stop_monitor")
        resp = await send("tools/call", {
            "name": "stop_monitor",
            "arguments": {"stream_id": stream_id},
        })
        print(f"    structured: {resp['result'].get('structuredContent')}")

        print("\nLIVE TEST OK")
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
