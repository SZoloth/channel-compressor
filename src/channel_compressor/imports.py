from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .utils import (
    dedupe_adjacent_segments,
    normalize_text,
    parse_video_id,
    read_jsonl,
    safe_float,
    sha256_text,
    transcript_text,
)
from .workspace import Workspace


def _records_from_path(path: Path) -> Iterable[tuple[dict[str, Any] | None, Path]]:
    paths = (
        sorted(
            item
            for pattern in ("*.json", "*.jsonl", "*.txt", "*.vtt", "*.srt")
            for item in path.rglob(pattern)
        )
        if path.is_dir()
        else [path]
    )
    for item_path in paths:
        try:
            if item_path.suffix.lower() == ".txt":
                yield {
                    "video_id": item_path.stem,
                    "text": item_path.read_text(encoding="utf-8"),
                    "source": "plain-text-export",
                }, item_path
                continue
            if item_path.suffix.lower() in {".vtt", ".srt"}:
                from .transcripts import parse_vtt

                yield {
                    "video_id": item_path.stem,
                    "segments": parse_vtt(item_path),
                    "source": f"{item_path.suffix.lower()[1:]}-export",
                }, item_path
                continue
            if item_path.suffix.lower() == ".jsonl":
                for record in read_jsonl(item_path):
                    yield record, item_path
                continue
            value = json.loads(item_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            yield None, item_path
            continue
        if isinstance(value, dict):
            yield value, item_path
        elif isinstance(value, list):
            # A bare list from transcript tools is usually the segment list for
            # the video named by the file. A list of records is also accepted.
            identifier_fields = {"video_id", "id", "url", "source_url"}
            is_segment_list = bool(value) and all(
                isinstance(row, dict)
                and row.get("text")
                and not identifier_fields.intersection(row)
                for row in value
            )
            if is_segment_list:
                yield {"video_id": item_path.stem, "segments": value}, item_path
            else:
                for record in value:
                    if isinstance(record, dict):
                        yield record, item_path
                    else:
                        yield None, item_path
        else:
            yield None, item_path


def _video_id(record: dict[str, Any], path: Path) -> str | None:
    for value in (
        record.get("video_id"),
        record.get("id"),
        record.get("url"),
        record.get("source_url"),
        path.stem,
    ):
        if value and (parsed := parse_video_id(str(value))):
            return parsed
    return None


def _segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = record.get("segments")
    if not isinstance(raw_segments, list) and isinstance(record.get("transcript"), list):
        raw_segments = record["transcript"]
    segments: list[dict[str, Any]] = []
    if isinstance(raw_segments, list):
        for raw in raw_segments:
            if isinstance(raw, str):
                segments.append({"text": raw, "start": 0.0, "duration": 0.0})
            elif isinstance(raw, dict) and raw.get("text"):
                segments.append(
                    {
                        "text": str(raw["text"]),
                        "start": safe_float(raw.get("start", raw.get("offset", 0.0))),
                        "duration": safe_float(raw.get("duration", 0.0)),
                    }
                )
    if not segments:
        raw_text = record.get("text") or record.get("transcript") or ""
        text = normalize_text(str(raw_text))
        if text:
            segments = [{"text": text, "start": 0.0, "duration": 0.0}]
    return dedupe_adjacent_segments(segments)


def import_transcripts(
    workspace: Workspace,
    source_path: Path,
    *,
    source_label: str = "external",
    overwrite: bool = False,
    minimum_words: int = 40,
) -> dict[str, int]:
    """Import transcript JSON without weakening provenance or corpus coverage checks.

    Supported inputs are canonical transcript objects, objects with ``id`` and a
    ``transcript`` string, segment arrays named by video ID, JSONL/list exports,
    and TXT/VTT/SRT files named by video ID. Only videos already present in the
    workspace manifest are accepted.
    """
    workspace.ensure()
    source_path = source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Transcript source does not exist: {source_path}")
    manifest = workspace.manifest_by_id()
    counts = {"imported": 0, "cached": 0, "empty": 0, "unmatched": 0, "invalid": 0}

    for record, record_path in _records_from_path(source_path):
        if record is None:
            counts["invalid"] += 1
            continue
        video_id = _video_id(record, record_path)
        if not video_id or video_id not in manifest:
            counts["unmatched"] += 1
            continue
        if workspace.load_transcript(video_id) and not overwrite:
            counts["cached"] += 1
            continue
        try:
            segments = _segments(record)
            text = transcript_text(segments)
        except (TypeError, ValueError):
            counts["invalid"] += 1
            continue
        if len(text.split()) < minimum_words:
            counts["empty"] += 1
            continue

        original_source = str(
            record.get("source") or record.get("transcript_kind") or "unspecified"
        )
        imported_at = datetime.now(timezone.utc).isoformat()
        workspace.save_transcript(
            video_id,
            {
                "video_id": video_id,
                "source": f"import:{normalize_text(source_label) or 'external'}",
                "segments": segments,
                "text": text,
                "word_count": len(text.split()),
                "text_sha256": sha256_text(text),
                "fetched_at": record.get("fetched_at") or imported_at,
                "imported_at": imported_at,
                "import_original_source": original_source,
                "import_file": record_path.name,
                "timestamps_available": any(
                    safe_float(item.get("start"), 0.0) > 0 for item in segments
                ),
            },
        )
        counts["imported"] += 1
    return counts
