# Validation status

Validated in the provided execution environment on 2026-09-03:

- Python source and tests compile successfully.
- Fourteen deterministic tests pass, including canonical, legacy, timestamped-segment, TXT, and
  short/unmatched external transcript-import cases.
- A synthetic end-to-end run successfully completed transcript loading, local analysis, semantic clustering, weighted selection, Markdown generation, HTML rendering, and CSV generation.
- CLI discovery and transcript integrations are covered with mocked provider behavior, including caption failure followed by `yt-dlp` success and cache reuse.
- Public-caption requests now have a configurable timeout so a single hung request cannot stall
  a channel-wide run.
- Reader retry behavior is covered for HTTP 429 responses, and Reader tag/note preservation is tested.
- A live `yt-dlp` inventory captured **183** public videos from `@erinmerylstudy/videos`, totaling
  about 32.9 hours. Public caption retrieval obtained **66** transcripts (134,651 words) before
  rate limits. The new importer brought all 66 into a canonical workspace; all 66 completed
  local analysis, while the other 117 remained explicitly marked as missing transcripts.
- The local selector was run with an 80% target, 120-minute budget, and 20%-of-eligible-videos
  cap. It selected 14 videos / 67.4 minutes and reached 21.9% of its weighted concept value before
  the fraction cap. The generated report correctly labeled this provisional and did not claim the
  80% target was met.

Not claimed:

- Full transcript coverage was not achieved: 117/183 videos were metadata-only after caption
  providers were rate-limited. The checked-in queue is therefore provisional for the whole
  channel, though every recommended video's transcript was available and reviewed.
- The 12-video queue is a human-reviewed evidence snapshot, not the output of the local lexical
  selector. A model-backed run should be used before treating automated semantic coverage as a
  robust final ranking.
- YouTube extraction interfaces are not stable contracts. The pipeline is intentionally resumable and uses multiple providers so a transient failure does not discard completed work.
