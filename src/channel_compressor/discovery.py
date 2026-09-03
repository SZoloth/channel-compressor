from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .utils import parse_video_id, safe_float
from .workspace import Workspace


def _canonical_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _video_record(entry: dict[str, Any], index: int) -> dict[str, Any] | None:
    raw_url = str(entry.get("webpage_url") or entry.get("url") or "")
    video_id = str(entry.get("id") or parse_video_id(raw_url) or "").strip()
    if not video_id:
        return None
    duration = entry.get("duration")
    duration_seconds = safe_float(duration, 0.0) or None
    live_status = entry.get("live_status")
    webpage_url = raw_url if parse_video_id(raw_url) else _canonical_watch_url(video_id)
    return {
        "id": video_id,
        "url": webpage_url,
        "title": entry.get("title") or video_id,
        "channel": entry.get("channel") or entry.get("uploader") or "",
        "channel_id": entry.get("channel_id") or entry.get("uploader_id") or "",
        "duration_seconds": duration_seconds,
        "upload_date": entry.get("upload_date") or "",
        "timestamp": entry.get("timestamp"),
        "view_count": entry.get("view_count"),
        "description": entry.get("description") or "",
        "playlist_index": entry.get("playlist_index") or index,
        "live_status": live_status,
        "is_live": live_status in {"is_live", "is_upcoming", "post_live"},
        "is_short": "/shorts/" in webpage_url,
        "availability": entry.get("availability") or "",
    }


def discover_channel(
    channel_url: str,
    workspace: Workspace,
    *,
    limit: int | None = None,
    include_lives: bool = False,
    include_shorts: bool = False,
) -> list[dict[str, Any]]:
    """Discover a channel tab using yt-dlp without downloading media."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is required for discovery. Install this project with `pip install -e .`."
        ) from exc

    workspace.ensure()
    existing = workspace.manifest_by_id()
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
        "lazy_playlist": False,
        "playlistend": limit,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    if not info:
        raise RuntimeError(f"yt-dlp returned no channel data for {channel_url}")

    entries = info.get("entries") or []
    discovered: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not entry:
            continue
        record = _video_record(dict(entry), index)
        if not record:
            continue
        if record["is_short"] and not include_shorts:
            continue
        if record["is_live"] and not include_lives:
            continue
        old = existing.get(record["id"], {})
        # Preserve richer prior metadata and processing state.
        merged = {**record, **{k: v for k, v in old.items() if v not in (None, "", [])}}
        # Fresh discovery metadata should win for volatile fields.
        for key in ("title", "duration_seconds", "view_count", "availability", "playlist_index"):
            if record.get(key) not in (None, ""):
                merged[key] = record[key]
        discovered.append(merged)
        if limit and len(discovered) >= limit:
            break

    if not discovered:
        raise RuntimeError(
            "No videos were discovered. Confirm the URL is a public channel tab such as `/videos`."
        )
    workspace.save_manifest(discovered)
    state = workspace.load_run_state()
    state.update(
        {
            "channel_url": channel_url,
            "channel_title": info.get("title") or info.get("channel") or "",
            "channel_id": info.get("channel_id") or "",
            "discovered_at": datetime.now(timezone.utc).isoformat(),
            "video_count": len(discovered),
        }
    )
    workspace.save_run_state(state)
    return discovered
