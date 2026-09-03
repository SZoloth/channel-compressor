import json
from pathlib import Path

from channel_compressor.imports import import_transcripts
from channel_compressor.workspace import Workspace


def _workspace(path: Path) -> Workspace:
    workspace = Workspace(path).ensure()
    workspace.save_manifest(
        [
            {
                "id": "aaaaaaaaaaa",
                "title": "One",
                "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                "playlist_index": 1,
            },
            {
                "id": "bbbbbbbbbbb",
                "title": "Two",
                "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
                "playlist_index": 2,
            },
        ]
    )
    return workspace


def test_imports_legacy_and_segment_transcript_exports(tmp_path: Path):
    workspace = _workspace(tmp_path / "workspace")
    source = tmp_path / "exports"
    source.mkdir()
    (source / "legacy.json").write_text(
        json.dumps(
            {
                "id": "aaaaaaaaaaa",
                "transcript": " ".join(["useful"] * 45),
                "transcript_kind": "web-caption-tool",
            }
        ),
        encoding="utf-8",
    )
    (source / "bbbbbbbbbbb.json").write_text(
        json.dumps(
            [
                {"text": "alpha " * 25, "start": 0, "duration": 4},
                {"text": "beta " * 25, "start": 4, "duration": 4},
            ]
        ),
        encoding="utf-8",
    )

    first = import_transcripts(workspace, source, source_label="saved-export")
    second = import_transcripts(workspace, source, source_label="saved-export")

    assert first == {"imported": 2, "cached": 0, "empty": 0, "unmatched": 0, "invalid": 0}
    assert second == {"imported": 0, "cached": 2, "empty": 0, "unmatched": 0, "invalid": 0}
    legacy = workspace.load_transcript("aaaaaaaaaaa")
    assert legacy["source"] == "import:saved-export"
    assert legacy["import_original_source"] == "web-caption-tool"
    assert legacy["timestamps_available"] is False
    segmented = workspace.load_transcript("bbbbbbbbbbb")
    assert segmented["timestamps_available"] is True
    assert segmented["word_count"] == 50


def test_import_rejects_unknown_and_short_records(tmp_path: Path):
    workspace = _workspace(tmp_path / "workspace")
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps(
            [
                {"id": "ccccccccccc", "transcript": "long " * 50},
                {"id": "aaaaaaaaaaa", "transcript": "too short"},
            ]
        ),
        encoding="utf-8",
    )

    counts = import_transcripts(workspace, source)

    assert counts == {"imported": 0, "cached": 0, "empty": 1, "unmatched": 1, "invalid": 0}


def test_bad_file_does_not_block_valid_list_export(tmp_path: Path):
    workspace = _workspace(tmp_path / "workspace")
    source = tmp_path / "exports"
    source.mkdir()
    (source / "bad.json").write_text("{not json", encoding="utf-8")
    (source / "records.json").write_text(
        json.dumps(
            [
                {
                    "video_id": "aaaaaaaaaaa",
                    "text": "valid " * 45,
                    "source": "another-tool",
                }
            ]
        ),
        encoding="utf-8",
    )

    counts = import_transcripts(workspace, source)

    assert counts == {"imported": 1, "cached": 0, "empty": 0, "unmatched": 0, "invalid": 1}


def test_imports_plain_text_named_by_video_id(tmp_path: Path):
    workspace = _workspace(tmp_path / "workspace")
    source = tmp_path / "aaaaaaaaaaa.txt"
    source.write_text("portable transcript " * 25, encoding="utf-8")

    counts = import_transcripts(workspace, source, source_label="browser-export")

    assert counts["imported"] == 1
    payload = workspace.load_transcript("aaaaaaaaaaa")
    assert payload["source"] == "import:browser-export"
    assert payload["import_original_source"] == "plain-text-export"
