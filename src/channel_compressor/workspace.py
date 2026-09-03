from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import (
    append_jsonl,
    atomic_write_json,
    load_json,
    read_jsonl,
    write_jsonl,
)


@dataclass(frozen=True)
class Workspace:
    root: Path

    def ensure(self) -> "Workspace":
        self.root.mkdir(parents=True, exist_ok=True)
        for child in ("transcripts", "analyses", "cache", "outputs"):
            (self.root / child).mkdir(parents=True, exist_ok=True)
        return self

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def errors_path(self) -> Path:
        return self.root / "errors.jsonl"

    @property
    def run_state_path(self) -> Path:
        return self.root / "run.json"

    @property
    def concepts_path(self) -> Path:
        return self.root / "concepts.jsonl"

    @property
    def clusters_path(self) -> Path:
        return self.root / "clusters.json"

    @property
    def selection_path(self) -> Path:
        return self.root / "selection.json"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    def transcript_path(self, video_id: str) -> Path:
        return self.root / "transcripts" / f"{video_id}.json"

    def analysis_path(self, video_id: str) -> Path:
        return self.root / "analyses" / f"{video_id}.json"

    def cache_dir(self, name: str) -> Path:
        path = self.root / "cache" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_manifest(self) -> list[dict[str, Any]]:
        return read_jsonl(self.manifest_path)

    def manifest_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(item["id"]): item for item in self.load_manifest() if item.get("id")}

    def save_manifest(self, videos: Iterable[dict[str, Any]]) -> None:
        ordered = sorted(videos, key=lambda row: (row.get("playlist_index") or 10**9, row.get("id", "")))
        write_jsonl(self.manifest_path, ordered)

    def save_transcript(self, video_id: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.transcript_path(video_id), payload)

    def load_transcript(self, video_id: str) -> dict[str, Any] | None:
        return load_json(self.transcript_path(video_id))

    def save_analysis(self, video_id: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self.analysis_path(video_id), payload)

    def load_analysis(self, video_id: str) -> dict[str, Any] | None:
        return load_json(self.analysis_path(video_id))

    def append_error(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.errors_path, payload)

    def save_run_state(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.run_state_path, payload)

    def load_run_state(self) -> dict[str, Any]:
        return load_json(self.run_state_path, default={}) or {}
