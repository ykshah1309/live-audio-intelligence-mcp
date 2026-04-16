# Contributing to live-audio-intelligence-mcp

Thanks for considering a contribution. Bug reports, calibration data, and
PRs are all welcome.

## Ground rules

- Keep the dependency footprint small. This package runs on analyst laptops;
  pulling in a 500 MB GPU wheel to save 10 lines of DSP code is not worth it.
- Don't add features that require a network call at import time. The MCP
  lifespan is already slow enough booting Whisper.
- Don't marketing-ify the docs. The stress score is a heuristic — anywhere
  it's described, it should be described honestly.

## Dev setup

```bash
git clone https://github.com/ykshah1309/live-audio-intelligence-mcp
cd live-audio-intelligence-mcp
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

You'll also need ffmpeg on PATH — see the README install section.

## Running checks before you open a PR

```bash
pytest -q                              # unit tests
python scripts/validate_stress_score.py  # synthetic-audio calibration
python scripts/smoke_test.py            # end-to-end smoke (needs ffmpeg + network)
```

If you're changing the stress-score heuristic (weights, saturation points,
feature set), the calibration benchmark must still pass — and you should
document the before/after numbers in the PR description.

## Filing bugs

Useful bug reports include:

- OS + Python version + ffmpeg version (`ffmpeg -version`).
- The URL you were monitoring (or a comparable public one).
- The full stderr output — the server logs aggressively to stderr with
  `[AudioStreamer]` / `[Transcriber]` / `[ProsodyAnalyzer]` tags.
- Whether `disable_vad=true` changes the behavior.

If the bug is a bad stress score on a specific recording, please attach
the audio (or a public URL to it). The score is a heuristic and "number
looks wrong" reports without audio can't be acted on.

## Scope of the stress score

This server's stress score is a **calibrated heuristic**, not a trained
model. We're not looking for PRs that ship pre-trained classifiers, and we
will not add dependencies on cloud inference APIs. Improvements to the
heuristic (better pause detection, better jitter estimation, better
handling of coded / compressed audio) are in scope. A full labeled-data
re-fit of the weights would be welcome but needs to come with the dataset.

## Commit + PR style

- One logical change per PR. Refactors go in their own PR.
- Keep commit messages short, imperative, and specific ("bump URL
  validator to reject file:// schemes", not "misc fixes").
- Rebase onto `main` before requesting review.
- New public functions / tools need a docstring that answers "what does
  this return and when would I call it?".

## Releases

Releases are cut from `main` after all checks pass:

1. Bump version in `pyproject.toml`, `src/live_audio_intelligence_mcp/__init__.py`,
   and `server.json`.
2. Add a section to `CHANGELOG.md`.
3. Tag: `git tag vX.Y.Z && git push --tags`.
4. Build: `python -m build`.
5. Upload: `twine upload dist/*`.
6. Publish to MCP registry: `mcp-publisher publish`.
7. `gh release create vX.Y.Z dist/*`.

## License

By contributing you agree that your contributions will be licensed under
the MIT License (see [LICENSE](LICENSE)).
