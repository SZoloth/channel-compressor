from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .utils import (
    dedupe_adjacent_segments,
    normalize_text,
    sha256_text,
    transcript_text,
)
from .workspace import Workspace

TIME_LINE_RE = re.compile(
    r"(?:(?P<h1>\d+):)?(?P<m1>\d{2}):(?P<s1>\d{2}[.,]\d{3})\s+-->\s+"
    r"(?:(?P<h2>\d+):)?(?P<m2>\d{2}):(?P<s2>\d{2}[.,]\d{3})"
)
CUE_TAG_RE = re.compile(r"<[^>]+>")


def _to_seconds(hours: str | None, minutes: str, seconds: str) -> float:
    return int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds.replace(",", "."))


def parse_vtt(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = TIME_LINE_RE.search(lines[index])
        if not match:
            index += 1
            continue
        start = _to_seconds(match.group("h1"), match.group("m1"), match.group("s1"))
        end = _to_seconds(match.group("h2"), match.group("m2"), match.group("s2"))
        index += 1
        cue_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            line = lines[index]
            if not TIME_LINE_RE.search(line):
                cue_lines.append(line)
            index += 1
        text = normalize_text(CUE_TAG_RE.sub(" ", " ".join(cue_lines)))
        if text:
            segments.append({"text": text, "start": start, "duration": max(0.0, end - start)})
        index += 1
    return dedupe_adjacent_segments(segments)


def _payload(
    video_id: str,
    source: str,
    segments: list[dict[str, Any]],
    **metadata: Any,
) -> dict[str, Any]:
    cleaned = dedupe_adjacent_segments(segments)
    text = transcript_text(cleaned)
    return {
        "video_id": video_id,
        "source": source,
        "segments": cleaned,
        "text": text,
        "word_count": len(text.split()),
        "text_sha256": sha256_text(text),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }


def fetch_youtube_captions(video_id: str, languages: list[str]) -> dict[str, Any]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise RuntimeError("youtube-transcript-api is not installed") from exc

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    transcript = transcript_list.find_transcript(languages)
    fetched = transcript.fetch()
    raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
    segments = [
        {
            "text": item["text"] if isinstance(item, dict) else item.text,
            "start": item.get("start", 0.0) if isinstance(item, dict) else item.start,
            "duration": item.get("duration", 0.0) if isinstance(item, dict) else item.duration,
        }
        for item in raw
    ]
    return _payload(
        video_id,
        "youtube-transcript-api",
        segments,
        language=getattr(fetched, "language", getattr(transcript, "language", "")),
        language_code=getattr(
            fetched, "language_code", getattr(transcript, "language_code", "")
        ),
        is_generated=bool(getattr(fetched, "is_generated", getattr(transcript, "is_generated", False))),
    )


def fetch_ytdlp_subtitles(
    video_id: str,
    url: str,
    cache_dir: Path,
    languages: list[str],
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    target_dir = cache_dir / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(target_dir / f"{video_id}.%(ext)s")
    lang_expression = ",".join(languages + [f"{language}.*" for language in languages])
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--ignore-config",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        lang_expression,
        "--sub-format",
        "vtt",
        "--no-playlist",
        "--output",
        output_template,
    ]
    browser = os.getenv("YTDLP_COOKIES_BROWSER", "").strip()
    if browser:
        command.extend(["--cookies-from-browser", browser])
    command.append(url)
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    files = sorted(target_dir.glob("*.vtt"), key=lambda item: ("auto" in item.name, item.name))
    if not files:
        detail = (completed.stderr or completed.stdout or "No VTT file produced").strip()[-1200:]
        raise RuntimeError(f"yt-dlp subtitle extraction failed: {detail}")
    segments = parse_vtt(files[0])
    if not segments:
        raise RuntimeError(f"Subtitle file was empty: {files[0].name}")
    return _payload(
        video_id,
        "yt-dlp-subtitles",
        segments,
        subtitle_file=files[0].name,
    )


_WHISPER_MODELS: dict[tuple[str, str, str], Any] = {}


def fetch_whisper(
    video_id: str,
    url: str,
    cache_dir: Path,
    *,
    model_name: str = "small.en",
    device: str = "cpu",
    compute_type: str = "int8",
) -> dict[str, Any]:
    try:
        import yt_dlp
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Whisper fallback is not installed. Install with `pip install -e '.[whisper]'`."
        ) from exc

    key = (model_name, device, compute_type)
    model = _WHISPER_MODELS.get(key)
    if model is None:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _WHISPER_MODELS[key] = model

    audio_dir = cache_dir / video_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    browser = os.getenv("YTDLP_COOKIES_BROWSER", "").strip()
    if browser:
        options["cookiesfrombrowser"] = (browser,)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        expected = Path(ydl.prepare_filename(info))
    audio_files = [expected] if expected.exists() else sorted(audio_dir.glob(f"{video_id}.*"))
    if not audio_files:
        raise RuntimeError("yt-dlp did not produce an audio file")

    generated, info = model.transcribe(
        str(audio_files[0]),
        beam_size=5,
        vad_filter=True,
        language="en" if model_name.endswith(".en") else None,
    )
    segments = [
        {
            "text": item.text,
            "start": float(item.start),
            "duration": max(0.0, float(item.end) - float(item.start)),
        }
        for item in generated
    ]
    if not segments:
        raise RuntimeError("Whisper found no speech")
    return _payload(
        video_id,
        "faster-whisper",
        segments,
        language=getattr(info, "language", ""),
        language_probability=getattr(info, "language_probability", None),
        model=model_name,
    )


def transcribe_workspace(
    workspace: Workspace,
    *,
    providers: list[str],
    languages: list[str] | None = None,
    force: bool = False,
    limit: int | None = None,
    delay_seconds: float = 0.4,
    whisper_model: str = "small.en",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    on_progress: Callable[[int, int, str, str], None] | None = None,
) -> dict[str, int]:
    workspace.ensure()
    videos = workspace.load_manifest()
    if limit:
        videos = videos[:limit]
    languages = languages or ["en", "en-US", "en-GB"]
    cache_dir = workspace.cache_dir("transcript-providers")
    counts = {"transcribed": 0, "cached": 0, "failed": 0}

    for index, video in enumerate(videos, start=1):
        video_id = str(video["id"])
        existing = workspace.load_transcript(video_id)
        if existing and existing.get("text") and not force:
            counts["cached"] += 1
            if on_progress:
                on_progress(index, len(videos), video_id, "cached")
            continue

        errors: list[str] = []
        result: dict[str, Any] | None = None
        for provider in providers:
            provider = provider.strip().lower()
            try:
                if provider in {"youtube", "youtube-transcript-api"}:
                    result = fetch_youtube_captions(video_id, languages)
                elif provider in {"ytdlp", "yt-dlp", "subtitles"}:
                    result = fetch_ytdlp_subtitles(
                        video_id, str(video["url"]), cache_dir, languages
                    )
                elif provider in {"whisper", "faster-whisper"}:
                    result = fetch_whisper(
                        video_id,
                        str(video["url"]),
                        cache_dir,
                        model_name=whisper_model,
                        device=whisper_device,
                        compute_type=whisper_compute_type,
                    )
                elif provider == "readwise":
                    cached_reader = workspace.root / "cache" / "readwise" / f"{video_id}.json"
                    if not cached_reader.exists():
                        raise RuntimeError(
                            "No cached Reader transcript; run `channel-compressor reader-fetch` first"
                        )
                    result = json.loads(cached_reader.read_text(encoding="utf-8"))
                else:
                    raise ValueError(f"Unknown transcript provider: {provider}")
                if result and result.get("text"):
                    result["provider_attempts"] = errors + [f"{provider}:success"]
                    break
            except Exception as exc:  # continue through explicit fallback chain
                errors.append(f"{provider}:{type(exc).__name__}:{exc}")
                result = None

        if result:
            workspace.save_transcript(video_id, result)
            counts["transcribed"] += 1
            status = str(result.get("source", "transcribed"))
        else:
            counts["failed"] += 1
            status = "failed"
            workspace.append_error(
                {
                    "stage": "transcript",
                    "video_id": video_id,
                    "title": video.get("title"),
                    "url": video.get("url"),
                    "errors": errors,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
        if on_progress:
            on_progress(index, len(videos), video_id, status)
        if delay_seconds > 0 and index < len(videos):
            time.sleep(delay_seconds)
    return counts
