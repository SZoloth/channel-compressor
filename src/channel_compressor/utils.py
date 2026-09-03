from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import parse_qs, urlparse

CTA_PATTERNS = (
    r"\bsubscribe\b",
    r"\bthumbs? up\b",
    r"\blike (?:this|the) video\b",
    r"\blet me know in the comments\b",
    r"\bfollow me\b",
    r"\bsee you in the next video\b",
    r"\bif you(?:'re| are) new here\b",
)
CTA_RE = re.compile("|".join(CTA_PATTERNS), re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
BRACKET_NOISE_RE = re.compile(
    r"\[(?:music|applause|laughter|laughs|clears throat|snorts?|inaudible)\]",
    re.IGNORECASE,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_json(path: Path, value: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=indent) + "\n")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = BRACKET_NOISE_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&quot;", '"')
    return SPACE_RE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def strip_cta_sentences(text: str) -> str:
    sentences = split_sentences(text)
    kept = [sentence for sentence in sentences if not CTA_RE.search(sentence)]
    return " ".join(kept)


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = SENTENCE_RE.split(cleaned)
    if len(parts) == 1:
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def dedupe_adjacent_segments(
    segments: Sequence[dict[str, Any]], similarity_cutoff: float = 0.92
) -> list[dict[str, Any]]:
    """Remove exact and near-exact adjacent caption repeats.

    Auto-captions frequently repeat a rolling fragment across consecutive cues.
    This intentionally handles only adjacent duplication; semantic de-duplication
    belongs in the corpus analysis stage.
    """
    from difflib import SequenceMatcher

    output: list[dict[str, Any]] = []
    previous = ""
    for raw in segments:
        text = normalize_text(str(raw.get("text", "")))
        if not text:
            continue
        comparable = normalize_for_match(text)
        if previous:
            ratio = SequenceMatcher(None, previous, comparable).ratio()
            if comparable == previous or ratio >= similarity_cutoff:
                continue
            # Rolling captions often prepend the previous cue verbatim.
            if comparable.startswith(previous) and len(comparable) > len(previous):
                previous_words = len(previous.split())
                words = text.split()
                text = " ".join(words[previous_words:]).strip() or text
                comparable = normalize_for_match(text)
        item = dict(raw)
        item["text"] = text
        output.append(item)
        previous = comparable
    return output


def transcript_text(segments: Sequence[dict[str, Any]]) -> str:
    return normalize_text(" ".join(str(item.get("text", "")) for item in segments))


def parse_video_id(value: str) -> str | None:
    value = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value
    parsed = urlparse(value)
    host = parsed.netloc.lower().removeprefix("www.")
    if host in {"youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [None])[0]
            return candidate if candidate and re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
        pieces = [piece for piece in parsed.path.split("/") if piece]
        if len(pieces) >= 2 and pieces[0] in {"shorts", "embed", "live"}:
            candidate = pieces[1]
            return candidate if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate) else None
    return None


def seconds_to_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "?"
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def timestamped_text(
    segments: Sequence[dict[str, Any]], marker_interval_seconds: int = 30
) -> str:
    parts: list[str] = []
    next_marker = 0.0
    for item in segments:
        start = float(item.get("start", 0.0) or 0.0)
        text = normalize_text(str(item.get("text", "")))
        if not text:
            continue
        if start >= next_marker:
            parts.append(f"[{seconds_to_timestamp(start)}]")
            next_marker = start + marker_interval_seconds
        parts.append(text)
    return " ".join(parts)


def chunk_text(text: str, max_chars: int = 14_000, overlap_chars: int = 700) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        if end < length:
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= length:
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse an LLM response even when it wraps JSON in Markdown prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return value
                break
    raise ValueError("Could not parse a JSON object from model response")
