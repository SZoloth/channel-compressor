# Agent guide

## Mission

Channel Compressor finds the smallest set of videos that captures most of a public YouTube channel's distinct value for a particular viewer. It is not a popularity ranker. It is a resumable evidence pipeline: inventory → transcripts → atomic concepts → semantic clusters → marginal-value selection → report.

## Repository map

- `src/channel_compressor/discovery.py` — channel inventory via `yt-dlp`.
- `src/channel_compressor/transcripts.py` — transcript provider cascade and caching.
- `src/channel_compressor/analysis.py` — local/model-backed analysis, clustering, and weighted set cover.
- `src/channel_compressor/llm.py` — OpenAI structured-output adapter.
- `src/channel_compressor/readwise.py` — optional Reader import, transcript cache, and shortlist sync.
- `src/channel_compressor/report.py` — Markdown, HTML, and CSV outputs.
- `src/channel_compressor/workspace.py` — stable workspace paths and persistence.
- `profiles/` — viewer-specific value functions.
- `tests/` — deterministic behavior and regression tests.

## Non-negotiable invariants

1. Never label a result complete when any discovered video lacks a transcript or analysis. Coverage must be described as coverage of the analyzed subset.
2. Re-runs must preserve completed work. Cache invalidation should be scoped to changed transcripts, profiles, models, or analysis modes.
3. Transcript providers fail independently. One provider failure must not abort the corpus.
4. Distinguish recommendation usefulness from evidentiary support. A practical tactic may survive while an unsupported causal explanation is penalized.
5. Selection is based on marginal semantic value per minute, not views, likes, or a fixed top-N.
6. Reader sync must preserve the user's existing tags and notes and replace only the marked Channel Compressor block.
7. Reports should paraphrase; do not redistribute full creator transcripts.
8. Keep the local analysis path functional without paid APIs.

## Development commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check src tests
python -m pytest
```

Smoke test without an API key:

```bash
channel-compressor run \
  'https://www.youtube.com/@erinmerylstudy/videos' \
  --workspace /tmp/channel-compressor-smoke \
  --profile profiles/sam.yaml \
  --analysis-mode local \
  --providers youtube,ytdlp \
  --limit 10
```

## Change discipline

- Add or update tests for behavior changes.
- Prefer small, inspectable JSON artifacts over hidden state.
- Keep external-service code behind narrow adapters.
- Do not add a new dependency when a standard-library implementation is adequate.
- Treat YouTube extraction as volatile and error messages as user-facing diagnostics.
