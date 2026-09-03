from pathlib import Path
from types import SimpleNamespace
import sys

from channel_compressor.discovery import discover_channel
from channel_compressor.workspace import Workspace


class _FakeYDL:
    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=False):
        assert download is False
        return {
            "title": "Mock Channel",
            "channel_id": "UC-mock",
            "entries": [
                {
                    "id": "aaaaaaaaaaa",
                    "title": "Useful long-form video",
                    "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                    "duration": 600,
                    "playlist_index": 1,
                    "live_status": "not_live",
                },
                {
                    "id": "bbbbbbbbbbb",
                    "title": "Short",
                    "webpage_url": "https://www.youtube.com/shorts/bbbbbbbbbbb",
                    "duration": 45,
                    "playlist_index": 2,
                    "live_status": "not_live",
                },
                {
                    "id": "ccccccccccc",
                    "title": "Livestream",
                    "url": "https://www.youtube.com/watch?v=ccccccccccc",
                    "duration": 3600,
                    "playlist_index": 3,
                    "live_status": "is_live",
                },
            ],
        }


def test_discovery_filters_and_persists(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=_FakeYDL))
    workspace = Workspace(tmp_path).ensure()

    videos = discover_channel("https://example.test/channel/videos", workspace)

    assert [video["id"] for video in videos] == ["aaaaaaaaaaa"]
    assert workspace.load_run_state()["channel_title"] == "Mock Channel"
    assert workspace.load_manifest()[0]["duration_seconds"] == 600.0
