# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/spec/v2.0.0.html).

## [0.1.3] — 2026-04-16

### Added
- Pytest unit suite under `tests/` covering URL syntactic validation,
  concurrency-cap enforcement, the custom exception hierarchy, and the
  prosody analyzer on synthetic audio (sine tone, silence, jittered pitch).
- `scripts/validate_stress_score.py` — synthetic-audio calibration harness.
  Generates controlled audio with known acoustic properties (smooth sine,
  jittered pitch, silence-padded speech) and verifies the stress score
  responds in the expected direction. Calibration evidence, not
  market-outcome validation.
- `CONTRIBUTING.md` and `CHANGELOG.md`.
- `[project.optional-dependencies]` `dev` group with `pytest`, `build`,
  and `twine`.
- README section on FFmpeg system install for macOS, Linux (Debian /
  Fedora), and Windows (winget / choco / scoop).
- README troubleshooting section.

### Notes
- **Not** included: empirical validation of stress-score weights against
  labeled market outcomes. That requires a labeled dataset of earnings
  calls tagged with subsequent price moves, which the maintainer does
  not have. The synthetic calibration harness is a directional sanity
  check, not a substitute.

## [0.1.2] — 2026-04-16

### Added
- Custom exception hierarchy (`LiveAudioError`, `InvalidStreamURLError`,
  `StreamLimitExceededError`, `StreamStartError`, `UnknownStreamError`)
  multi-inheriting from `ValueError` / `RuntimeError` for backward
  compatibility with clients that catch the builtins.
- Syntactic URL validation (`_validate_url_syntax`): rejects non-`http(s)`
  schemes, empty strings, and URLs without a host before any subprocess
  or network call. Blocks `file://`, `javascript:`, `data:`, etc.
- Concurrent-stream cap (default 4) with env override
  `LAI_MAX_CONCURRENT_STREAMS`. Exceeding the cap raises
  `StreamLimitExceededError` instead of silently queuing.

### Changed
- README stripped of "institutional-grade" / "alpha generator" hype.
  Stress score is now documented as a heuristic with hand-picked weights,
  with explicit saturation points (jitter = 0.12, hesitation = 0.30)
  and a note that weights are not fit to any labeled dataset.
- `pyproject.toml` sdist now has an explicit `include` list. Previous
  versions accidentally packaged the local `tools/` directory, inflating
  the sdist to 7.2 MB; v0.1.2 sdist is 23 KB.

### Fixed
- Broken `git clone` URL in README (`live-audio-intelligence/...` →
  `ykshah1309/...`).
- `scripts/smoke_test.py` and `scripts/live_test.py` now use
  `sys.executable` instead of a hard-coded Windows `.venv/Scripts/python.exe`.
- Removed unused `ffmpeg-python` dependency from `requirements.txt`.

## [0.1.1] — 2026-04-16

### Added
- `mcp-name` marker in README required for PyPI ownership validation
  by the MCP registry.

### Changed
- MCP registry manifest schema upgraded from `2025-09-29` (deprecated)
  to `2025-12-11`. Added `registryBaseUrl: https://pypi.org` for the
  PyPI package entry.

## [0.1.0] — 2026-04-16

### Added
- Initial release. FastMCP stdio server with four tools:
  `monitor_live_stream`, `get_rolling_transcript`,
  `analyze_speaker_stress`, `stop_monitor`.
- yt-dlp → ffmpeg piped ingestion for any webcast URL.
- `faster-whisper base.en` int8 CPU transcription.
- pYIN-based prosody analyzer with composite 0–100 stress score.

[0.1.3]: https://github.com/ykshah1309/live-audio-intelligence-mcp/releases/tag/v0.1.3
[0.1.2]: https://github.com/ykshah1309/live-audio-intelligence-mcp/releases/tag/v0.1.2
[0.1.1]: https://github.com/ykshah1309/live-audio-intelligence-mcp/releases/tag/v0.1.1
[0.1.0]: https://github.com/ykshah1309/live-audio-intelligence-mcp/releases/tag/v0.1.0
