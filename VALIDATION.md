# Validation — September 7, 2026

**Ready to share with explicit caveats.** The final deliverable is [the practical guide](examples/erin_meryl_final_80_20.md), not a validated 80% semantic result.

## Dataset and checks

- September 3 inventory: 183 unique video IDs, 1,974.93 source minutes (32.92 hours).
- September 4 transcript recovery: 177 nonempty transcripts, 373,402 words; 66 original caption imports plus 111 verified Reader imports.
- All 177 transcript hashes match their text and downstream analyses; all analyses have concepts and summaries. No exact duplicate transcript hashes. This does not prove captions are error-free.
- Six missing transcripts remain disclosed in the final guide. Their titles describe ambient study sessions and Reader returned no meaningful transcript. Their entire audio was not independently checked. Corpus completeness is **false**.
- All 177 available transcripts were reanalyzed locally September 7 with `tfidf-mmr-v2`: 177 analyzed, zero failed, six missing. Model versioning invalidates pre-fix local caches.
- Local selection: 24 videos, 119.5 minutes, 15.3% lexical weighted coverage, stopped at the time budget. This is a heuristic on the analyzed subset, not a semantic coverage estimate.
- Editorial route: six videos, 60.6167 minutes. Three optional videos add 26.4167 minutes. Durations checked against manifest seconds. No claim of globally optimal selection.
- Sixteen tests pass; Ruff passes for source, tests and audit script. Coverage includes provider fallback/caching, import, Reader retries and note preservation, punctuation-free captions, partial-report and lexical warnings, selection, and stale-analysis auditing. The full local run regenerated Markdown, HTML and queue outputs.

## Reproduce on the private corpus

```bash
uv run --extra dev pytest -q
uv run ruff check src tests scripts/audit_corpus.py
uv run channel-compressor analyze --workspace erin-meryl-corpus \
  --profile profiles/sam.yaml --mode local
python scripts/audit_corpus.py erin-meryl-corpus
```

The audit emits counts, missing public video IDs and integrity errors, not transcript text or Reader details. Exit zero means no integrity errors, **not** complete coverage: check its `complete` and `missing_transcript_ids` fields. Current result: `complete: false`, six missing IDs, no integrity errors.

## Publication boundaries and limitations

Only code, tests, public-video links and paraphrased findings are published. The corpus and Reader exports remain gitignored. Reproduction from scratch requires fetching your own transcripts; provider limits may change availability.

The final guide supersedes the dated September 3 and September 4 examples and CSVs, which remain historical. Review was assistant-led, not independent human or scientific review. Neural mechanisms, numerical effect sizes and visual demonstrations have not been independently validated. Later uploads are outside the September 3 inventory.
