from pathlib import Path

import channel_compressor.analysis as analysis_module
from channel_compressor.analysis import analyze_transcripts
from channel_compressor.utils import sha256_text
from channel_compressor.workspace import Workspace


class _FailingAnalyzer:
    model = "mock-model"

    def __init__(self, model=None):
        pass

    def analyze_video(self, **kwargs):
        raise RuntimeError("temporary API failure")


def test_auto_mode_falls_back_locally(monkeypatch, tmp_path: Path):
    workspace = Workspace(tmp_path).ensure()
    video_id = "aaaaaaaaaaa"
    text = (
        "Read one section and close the book. Explain the idea from memory, inspect errors, "
        "and schedule the next repetition."
    )
    workspace.save_manifest(
        [
            {
                "id": video_id,
                "title": "Retrieval protocol",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "playlist_index": 1,
            }
        ]
    )
    workspace.save_transcript(
        video_id,
        {
            "text": text,
            "segments": [{"text": text, "start": 0.0, "duration": 10.0}],
            "text_sha256": sha256_text(text),
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(analysis_module, "OpenAIAnalyzer", _FailingAnalyzer)

    counts = analyze_transcripts(workspace, profile_path=None, mode="auto")
    payload = workspace.load_analysis(video_id)

    assert counts["analyzed"] == 1
    assert payload["mode"] == "local"
    assert payload["fallback_from"] == "openai"
    assert payload["fallback_reason"] == "RuntimeError"
