# live-audio-intelligence-mcp

<!-- mcp-name: io.github.ykshah1309/live-audio-intelligence-mcp -->

**MCP server for live financial webcast transcription and heuristic vocal stress analysis.**

Turns any live webcast URL (earnings calls, CNBC, investor days) into a real-time
pipeline that feeds an LLM two things simultaneously:

1. A **rolling transcript** via `faster-whisper` (CPU, int8).
2. A **heuristic vocal stress score (0–100)** derived from F0 pitch jitter,
   hesitation ratio, and voiced-frame fraction. These prosodic features are
   well-established correlates of speaker arousal in the vocal-analysis
   literature; their composition into the score below is heuristic and has
   **not** been empirically validated against market outcomes. Treat it as a
   coarse signal, not an oracle.

Built on the [Model Context Protocol](https://modelcontextprotocol.io). Exposes
4 tools over stdio; drop it into Claude Desktop, Claude Code, or any MCP client.

---

## Why this exists

Sell-side analysts and hedge-fund PMs don't just want to read the earnings
transcript after the fact — they want a real-time signal about **how confident
the CFO sounds when asked about Q4 guidance**. This server wires a Whisper
pipeline and a pYIN-based prosody analyzer directly into an LLM's tool loop,
so the model can ask *"what did the CEO just say about China?"* and *"how
stressed did they sound saying it?"* in the same conversation.

---

## Install

Requires **Python ≥ 3.10** and **ffmpeg** on your PATH.

```bash
pip install live-audio-intelligence-mcp
```

Verify ffmpeg:

```bash
ffmpeg -version
```

The first run will download the `faster-whisper base.en` model (~140 MB).

---

## Run it

Stdio MCP server:

```bash
live-audio-intelligence-mcp
```

Or equivalently:

```bash
python -m live_audio_intelligence_mcp
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "live-audio-intelligence": {
      "command": "live-audio-intelligence-mcp"
    }
  }
}
```

### Claude Code

```bash
claude mcp add live-audio-intelligence -- live-audio-intelligence-mcp
```

---

## Tools

| Tool | Purpose |
|---|---|
| `monitor_live_stream(url, disable_vad=False)` | Resolve the audio URL, spawn ffmpeg, start chunking + transcription. Returns a `stream_id`. |
| `get_rolling_transcript(stream_id, minutes_back=10)` | Get the last N minutes of concatenated transcript text. |
| `analyze_speaker_stress(stream_id, time_window_seconds=60)` | Run prosody analysis over the last N seconds of audio. Returns stress score, pitch jitter, hesitation ratio, pause stats, and a human-readable interpretation. |
| `stop_monitor(stream_id)` | Kill ffmpeg, clean up temp files, drop the transcript buffer. |

### The stress score

| Score | Interpretation |
|---|---|
| 0–20 | Confident, fluent delivery |
| 20–45 | Normal variation |
| 45–75 | Elevated stress — worth monitoring |
| 75–100 | High stress — potential market-moving signal |

Composite of:
- **Pitch jitter** (coefficient of variation of F0) — 50% weight, saturating at jitter = 0.12
- **Hesitation ratio** (fraction of audio in pauses > 400 ms) — 35% weight, saturating at 0.30
- **Unvoiced fraction** (speaker trailing off) — 15% weight

The three features are literature-backed correlates of speaker arousal (see
pYIN for F0 tracking, and the broad "disfluency is a correlate of cognitive
load" line of work). The *weights* and *saturation points* are hand-picked
defaults, chosen so that a calm speaker scores in the 0–20 band on clean
studio audio and visibly stressed speech scores ≥ 45 — they are not fit to any
labeled dataset. Consumers who care about absolute numbers should recalibrate
thresholds against their own recordings.

### Low-SNR mode

For speakerphone audio (most earnings Q&A), pass `disable_vad=true` to
`monitor_live_stream`. Silero VAD tends to aggressively classify muddy
conference-call speech as silence; disabling it preserves more of the speech
at the cost of transcribing a bit more ambient noise.

---

## Architecture

```
                 ┌──────────────────┐
    URL  ─────▶  │  yt-dlp resolve  │
                 └────────┬─────────┘
                          │ audio URL
                          ▼
                 ┌──────────────────┐      ┌────────────────┐
                 │  ffmpeg (bg)     │ ───▶ │  15s WAV chunk │
                 │  16kHz mono PCM  │      │  queue         │
                 └──────────────────┘      └───────┬────────┘
                                                   │
                                ┌──────────────────┴────────────────┐
                                ▼                                   ▼
                       ┌──────────────────┐              ┌──────────────────┐
                       │ faster-whisper   │              │  librosa.pyin    │
                       │ (int8 / CPU)     │              │  + pause detect  │
                       └────────┬─────────┘              └────────┬─────────┘
                                │ rolling transcript              │ stress score
                                ▼                                 ▼
                            ┌────────────── MCP stdio ───────────────┐
                            │    LLM (Claude) — calls tools freely   │
                            └────────────────────────────────────────┘
```

All blocking work (Whisper inference, ffmpeg I/O, librosa DSP) is dispatched
to threads via `asyncio.to_thread` so the MCP event loop stays responsive.

---

## Development

```bash
git clone https://github.com/ykshah1309/live-audio-intelligence-mcp
cd live-audio-intelligence-mcp
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
live-audio-intelligence-mcp
```

---

## License

MIT — see [LICENSE](LICENSE).
