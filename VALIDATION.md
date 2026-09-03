# Validation status

Validated in the provided execution environment on 2026-09-02:

- Python source and tests compile successfully.
- Ten deterministic tests pass.
- A synthetic end-to-end run successfully completed transcript loading, local analysis, semantic clustering, weighted selection, Markdown generation, HTML rendering, and CSV generation.
- CLI discovery and transcript integrations are covered with mocked provider behavior, including caption failure followed by `yt-dlp` success and cache reuse.
- Reader retry behavior is covered for HTTP 429 responses, and Reader tag/note preservation is tested.

Not claimed:

- The live `@erinmerylstudy` channel was not crawled end-to-end inside this container. Outbound package installation and direct YouTube access were unavailable to the runtime. The included preliminary queue is based on 13 transcripts inspected through external retrieval, and is explicitly labeled provisional.
- YouTube extraction interfaces are not stable contracts. The pipeline is intentionally resumable and uses multiple providers so a transient failure does not discard completed work.
