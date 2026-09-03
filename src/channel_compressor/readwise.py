from __future__ import annotations

import html
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests
from bs4 import BeautifulSoup

from .utils import normalize_text, parse_video_id, sha256_text
from .workspace import Workspace


class ReaderClient:
    BASE = "https://readwise.io/api/v3"

    def __init__(self, token: str | None = None, *, timeout: int = 40) -> None:
        self.token = (token or os.getenv("READWISE_TOKEN") or "").strip()
        if not self.token:
            raise RuntimeError(
                "READWISE_TOKEN is not set. Put the token in your shell environment; do not commit it."
            )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {self.token}",
                "User-Agent": "channel-compressor/0.1 (+personal research workflow)",
            }
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        last_response: requests.Response | None = None
        for attempt in range(5):
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            last_response = response
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if not retryable or attempt == 4:
                response.raise_for_status()
                return response
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(30.0, 2.0 ** attempt)
            except ValueError:
                delay = min(30.0, 2.0 ** attempt)
            time.sleep(max(0.1, delay))
        assert last_response is not None
        last_response.raise_for_status()
        return last_response

    def save_video(
        self,
        url: str,
        *,
        tags: list[str],
        location: str = "archive",
        notes: str = "",
    ) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"{self.BASE}/save/",
            json={
                "url": url,
                "category": "video",
                "location": location,
                "tags": tags,
                "notes": notes,
                "saved_using": "channel-compressor",
            },
        )
        return dict(response.json())

    def list_videos(
        self,
        *,
        tag: str,
        with_html: bool = True,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: list[tuple[str, str]] = [
                ("category", "video"),
                ("tag", tag),
                ("limit", "100"),
                ("withHtmlContent", "true" if with_html else "false"),
            ]
            if cursor:
                params.append(("pageCursor", cursor))
            response = self._request("GET", f"{self.BASE}/list/", params=params)
            payload = response.json()
            results.extend(payload.get("results") or [])
            cursor = payload.get("nextPageCursor")
            if not cursor:
                break
        return results

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        response = self._request(
            "GET",
            f"{self.BASE}/list/",
            params={"id": document_id, "limit": "1", "withHtmlContent": "false"},
        )
        results = response.json().get("results") or []
        return dict(results[0]) if results else None

    def update(
        self,
        document_id: str,
        *,
        location: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if location is not None:
            body["location"] = location
        if tags is not None:
            body["tags"] = tags
        if notes is not None:
            body["notes"] = notes
        response = self._request(
            "PATCH",
            f"{self.BASE}/update/{document_id}",
            json=body,
        )
        return dict(response.json())


def import_manifest_to_reader(
    workspace: Workspace,
    *,
    tag: str = "channel-compressor-corpus",
    location: str = "archive",
    rate_per_minute: int = 45,
    force: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    client = ReaderClient()
    videos = workspace.load_manifest()
    state_path = workspace.root / "cache" / "readwise-imported.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    from .utils import atomic_write_json, load_json

    imported: dict[str, Any] = load_json(state_path, default={}) or {}
    counts = {"saved": 0, "cached": 0, "failed": 0}
    interval = 60.0 / max(1, min(rate_per_minute, 50))
    for index, video in enumerate(videos, start=1):
        video_id = str(video["id"])
        if video_id in imported and not force:
            counts["cached"] += 1
            if on_progress:
                on_progress(index, len(videos), "cached")
            continue
        try:
            response = client.save_video(
                str(video["url"]), tags=[tag], location=location
            )
            imported[video_id] = {
                "document_id": response.get("id"),
                "reader_url": response.get("url"),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(state_path, imported)
            counts["saved"] += 1
            status = "saved"
        except Exception as exc:
            workspace.append_error(
                {
                    "stage": "readwise-import",
                    "video_id": video_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            counts["failed"] += 1
            status = "failed"
        if on_progress:
            on_progress(index, len(videos), status)
        if index < len(videos):
            time.sleep(interval)
    return counts


def _reader_html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    text = soup.get_text(" ")
    return normalize_text(html.unescape(text))


def fetch_reader_transcripts(
    workspace: Workspace,
    *,
    tag: str = "channel-compressor-corpus",
    overwrite: bool = False,
) -> dict[str, int]:
    client = ReaderClient()
    documents = client.list_videos(tag=tag, with_html=True)
    cache_dir = workspace.root / "cache" / "readwise"
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {"fetched": 0, "empty": 0, "unmatched": 0, "cached": 0}
    from .utils import atomic_write_json

    for document in documents:
        source_url = str(
            document.get("source_url")
            or document.get("raw_source_url")
            or document.get("url")
            or ""
        )
        video_id = parse_video_id(source_url)
        if not video_id:
            counts["unmatched"] += 1
            continue
        path = cache_dir / f"{video_id}.json"
        if path.exists() and not overwrite:
            counts["cached"] += 1
            continue
        text = _reader_html_to_text(str(document.get("html_content") or ""))
        # Very short content is usually metadata while Reader is still processing.
        if len(text.split()) < 40:
            counts["empty"] += 1
            continue
        payload = {
            "video_id": video_id,
            "source": "readwise-reader",
            "segments": [{"text": text, "start": 0.0, "duration": 0.0}],
            "text": text,
            "word_count": len(text.split()),
            "text_sha256": sha256_text(text),
            "reader_document_id": document.get("id"),
            "reader_url": document.get("url"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(path, payload)
        counts["fetched"] += 1
    return counts


def _reader_tag_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        names: set[str] = set()
        for key, details in value.items():
            if isinstance(details, dict) and details.get("name"):
                names.add(str(details["name"]))
            elif key:
                names.add(str(key))
        return names
    if isinstance(value, list):
        return {str(item.get("name") if isinstance(item, dict) else item) for item in value if item}
    return set()


def _merge_channel_compressor_note(existing: str, generated: str) -> str:
    marker = "\n\n---\n[Channel Compressor selection] "
    base = (existing or "").split(marker, 1)[0].rstrip()
    return (base + marker + generated).strip()


def sync_selection_to_reader(
    workspace: Workspace,
    *,
    tag: str = "channel-compressor-watch",
    location: str = "later",
) -> dict[str, int]:
    from .utils import load_json

    selection = load_json(workspace.selection_path, default={}) or {}
    selected = selection.get("selected") or []
    manifest = workspace.manifest_by_id()
    client = ReaderClient()
    counts = {"synced": 0, "failed": 0}
    for item in selected:
        video_id = str(item["video_id"])
        video = manifest.get(video_id)
        if not video:
            continue
        analysis = workspace.load_analysis(video_id) or {}
        notes = normalize_text(
            f"Channel Compressor rank #{item.get('rank')}. "
            f"Coverage added: {float(item.get('marginal_share', 0)):.1%}. "
            f"{analysis.get('summary', '')}"
        )
        try:
            response = client.save_video(
                str(video["url"]), tags=[tag], location=location, notes=notes
            )
            document_id = response.get("id")
            if document_id:
                existing = client.get_document(str(document_id)) or {}
                merged_tags = sorted(_reader_tag_names(existing.get("tags")) | {tag})
                merged_notes = _merge_channel_compressor_note(str(existing.get("notes") or ""), notes)
                client.update(
                    str(document_id),
                    location=location,
                    tags=merged_tags,
                    notes=merged_notes,
                )
            counts["synced"] += 1
        except Exception as exc:
            workspace.append_error(
                {
                    "stage": "readwise-sync",
                    "video_id": video_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
            counts["failed"] += 1
    return counts
