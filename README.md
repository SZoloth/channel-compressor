# Channel Compressor

**Channel Compressor** answers a better question than “Which uploads are popular?”

> **What is the smallest set of videos that covers most of a channel’s distinct, useful value for a particular person?**

It inventories a YouTube channel, captures every obtainable transcript through an explicit fallback chain, extracts atomic ideas, clusters semantic repetition, discounts weakly supported explanatory claims, and selects videos by **weighted marginal value per minute** until it reaches a target such as 80%.

This copy includes a Sam-specific profile and defaults to `@erinmerylstudy/videos`, but the code does not hard-code the creator or a channel ID.

## The easiest path

On macOS or Linux:

```bash
git clone https://github.com/SZoloth/channel-compressor.git
cd channel-compressor
export OPENAI_API_KEY='...'       # optional, but materially improves the analysis
./scripts/run_erin_meryl.sh
```

The script creates a virtual environment, installs dependencies, discovers the channel, captures transcripts, analyzes them, and writes:

```text
erin-meryl-corpus/outputs/report.html
```

A ChatGPT subscription and OpenAI API billing are separate. Without an API key, Channel Compressor runs its local TF–IDF/SVD analysis instead.

## A more controlled install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[all]'

channel-compressor run \
  'https://www.youtube.com/@erinmerylstudy/videos' \
  --workspace erin-meryl-corpus \
  --profile profiles/sam.yaml \
  --target-coverage 0.80 \
  --max-minutes 120
```

The default provider order is:

```text
youtube-transcript-api → yt-dlp subtitles → faster-whisper
```

Reader is opt-in rather than silently included in the default chain.

## Why transcript capture needs fallbacks

YouTube’s official caption-download endpoint requires permission to edit the video, so it is not a general public-channel transcript API. Channel Compressor instead uses:

1. `youtube-transcript-api` for public manual or automatic captions;
2. `yt-dlp` subtitle extraction as an independent fallback;
3. cached Readwise Reader transcript HTML, when you deliberately import the corpus;
4. local `faster-whisper` transcription when no caption track exists.

Every successful result is cached by video ID and transcript hash. Failures go to `errors.jsonl`. Re-running is safe: completed work is retained, and changed transcripts invalidate only their downstream analysis.

YouTube extraction changes periodically. Keep `yt-dlp` current; its nightly build may recover sooner than the stable release after a YouTube change. Recent extraction also benefits from having a supported JavaScript runtime such as Deno or Node installed.

## Readwise Reader integration

Reader is best used as the **consumption and highlighting layer**, not as the only crawler. Saving every channel upload can also pollute your inbox, so Channel Compressor’s clean default is to process locally and sync only the final shortlist.

```bash
export READWISE_TOKEN='...'
channel-compressor reader-sync --workspace erin-meryl-corpus \
  --tag erin-meryl-80-20 --location later
```

Channel Compressor preserves existing Reader tags and user notes. It replaces only its own marked selection block.

To use Reader as a full-corpus transcript provider:

```bash
# Save all videos with one tag, archived rather than inboxed.
channel-compressor reader-import --workspace erin-meryl-corpus \
  --tag erin-meryl-corpus --location archive

# Fetch Reader's generated HTML content into the provider cache.
channel-compressor reader-fetch --workspace erin-meryl-corpus \
  --tag erin-meryl-corpus

# Then run or re-run transcript capture with Reader in the desired position.
channel-compressor transcribe --workspace erin-meryl-corpus \
  --providers readwise,youtube,ytdlp,whisper
```

Reader may still be processing a newly saved video. Very short HTML payloads are treated as incomplete rather than cached as fake transcripts.

Keep API tokens in shell environment variables. Do not put them in source files or commit `.env`.

## Staged workflow

The one-shot command is convenient, but the staged commands are better for inspection or recovery:

```bash
channel-compressor discover 'https://www.youtube.com/@erinmerylstudy/videos' \
  --workspace erin-meryl-corpus

channel-compressor transcribe --workspace erin-meryl-corpus

channel-compressor analyze --workspace erin-meryl-corpus \
  --profile profiles/sam.yaml \
  --mode auto \
  --target-coverage 0.80 \
  --max-minutes 120

channel-compressor report --workspace erin-meryl-corpus
channel-compressor status --workspace erin-meryl-corpus
```

Useful variants:

```bash
# Free, deterministic local analysis
channel-compressor analyze -w erin-meryl-corpus -p profiles/sam.yaml --mode local

# Captions only; do not download audio
channel-compressor transcribe -w erin-meryl-corpus --providers youtube,ytdlp

# Re-attempt failed/missing transcript work
channel-compressor transcribe -w erin-meryl-corpus --force

# Inspect a small slice before committing to the full corpus
channel-compressor run -w erin-meryl-smoke-test --limit 10 --analysis-mode local
```

## Outputs

`<workspace>/outputs/` contains:

- `report.md` — canonical narrative analysis;
- `report.html` — polished, self-contained browsing report;
- `ranked_videos.csv` — every eligible video with selection, novelty, and redundancy fields;
- `concept_clusters.csv` — repeated themes and verification risk;
- `readwise_queue.csv` — final queue in a portable import-friendly format.

Intermediate evidence remains inspectable:

- `manifest.jsonl` — channel inventory and metadata;
- `transcripts/<video_id>.json` — transcript, timestamps, provenance, hash, and provider attempts;
- `analyses/<video_id>.json` — neutral summary, atomic concepts, viewer fit, compressibility, and cautions;
- `concepts.jsonl`, `clusters.json`, `selection.json` — the de-duplication and set-cover calculation;
- `errors.jsonl` — append-only failures, preserving the rest of the run.

The report prominently says **provisional** unless every discovered video has both a transcript and an analysis. A displayed 80% then means 80% of the analyzed subset, never unseen material.

## What “80/20” means

It does **not** choose the top 20% by views. Each semantic concept cluster is treated as a unit of value. For each candidate, Channel Compressor asks what it contributes beyond the videos already selected:

```text
concept utility
  = salience
  × viewer relevance
  × actionability
  × epistemic quality
  × specificity/kind adjustments

marginal value
  = Σ(cluster importance × newly covered share)
  × audience-fit adjustment
  × verification penalty

selection score
  = marginal value / duration^0.78
```

The greedy selector stops at the first of:

- the requested weighted-coverage target;
- the time budget;
- the maximum allowed fraction of eligible videos;
- no remaining material value.

“Covered by the chosen set” is calculated against the union of selected videos. “Closest chosen video” is reported separately, preventing a video whose ideas are spread across several selections from appearing falsely novel.

## Epistemic scoring

Self-help media often blurs three different questions:

1. Is the recommendation concrete enough to test?
2. Is it useful to this viewer?
3. Does the transcript substantiate the causal or neuroscience explanation attached to it?

Channel Compressor scores those separately. A tactic can survive while its “dopamine” wrapper is downgraded. This is intentional: practical usefulness should not launder an unsupported mechanism into fact.

## Local versus model-backed analysis

`--mode openai` or `--mode auto` with `OPENAI_API_KEY` uses structured model output to produce clean paraphrases, distinguish examples from durable principles, judge Sam-specific relevance, and flag evidentiary overreach. Embeddings are then used to cluster semantically equivalent concepts.

`--mode local` uses sentence selection, lexical relevance, TF–IDF/SVD embeddings, and agglomerative clustering. It is free and useful for triage, but less reliable when two videos express the same idea with very different language.

Changing the profile, transcript, model, or analysis mode invalidates stale cached analyses automatically.

## Viewer profile

Edit `profiles/sam.yaml` to change what “value” means. The included profile emphasizes:

- durable learning and mastery;
- execution and follow-through;
- product strategy, entrepreneurship, AI, and technical growth;
- sleep and sustainable performance;
- concrete protocols over generic motivation;
- a penalty for exam-only tactics, passive ambience, and neuroscience theater.

This personalization is the point. A Cambridge student cramming for exams and a product strategist building a long-term curriculum should not receive the same 80/20 queue.

## Testing

```bash
PYTHONPATH=src python -m pytest
```

The ten-test deterministic suite covers channel filtering, transcript-provider fallback, resumability, Reader retry/state behavior, semantic selection, tiny corpora, Markdown escaping, CSV generation, and HTML rendering.

## Responsible use

Use Channel Compressor for personal research on public videos. Do not redistribute creators’ full transcripts. The report paraphrases ideas and links back to the source. Extraction interfaces can change; the provider cascade and inspectable cache are designed so one brittle dependency does not sink the corpus.
