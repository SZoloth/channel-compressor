from pathlib import Path

import channel_compressor.transcripts as transcript_module
from channel_compressor.transcripts import transcribe_workspace
from channel_compressor.workspace import Workspace


def test_transcript_fallback_is_resumable(monkeypatch, tmp_path: Path):
    workspace = Workspace(tmp_path).ensure()
    workspace.save_manifest(
        [
            {
                "id": "aaaaaaaaaaa",
                "title": "A",
                "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                "playlist_index": 1,
            }
        ]
    )

    def fail_youtube(video_id, languages):
        raise RuntimeError("captions unavailable")

    def succeed_ytdlp(video_id, url, cache_dir, languages):
        return {
            "video_id": video_id,
            "source": "yt-dlp-subtitles",
            "segments": [{"text": "A useful transcript.", "start": 0.0, "duration": 2.0}],
            "text": "A useful transcript.",
            "word_count": 3,
            "text_sha256": "hash",
        }

    monkeypatch.setattr(transcript_module, "fetch_youtube_captions", fail_youtube)
    monkeypatch.setattr(transcript_module, "fetch_ytdlp_subtitles", succeed_ytdlp)

    first = transcribe_workspace(
        workspace,
        providers=["youtube", "ytdlp", "whisper"],
        delay_seconds=0,
    )
    second = transcribe_workspace(
        workspace,
        providers=["youtube", "ytdlp", "whisper"],
        delay_seconds=0,
    )

    assert first == {"transcribed": 1, "cached": 0, "failed": 0}
    assert second == {"transcribed": 0, "cached": 1, "failed": 0}
    payload = workspace.load_transcript("aaaaaaaaaaa")
    assert payload["source"] == "yt-dlp-subtitles"
    assert payload["provider_attempts"][0].startswith("youtube:RuntimeError")
    assert payload["provider_attempts"][-1] == "ytdlp:success"
