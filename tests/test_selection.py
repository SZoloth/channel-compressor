from pathlib import Path

from channel_compressor.analysis import cluster_and_select
from channel_compressor.workspace import Workspace


def _analysis(video_id, concepts):
    return {
        "video_id": video_id,
        "mode": "local",
        "summary": video_id,
        "watchability": 0.3,
        "compressibility": 0.9,
        "audience_fit": 0.8,
        "concepts": [
            {
                "claim": claim,
                "kind": "method",
                "salience": 1.0,
                "actionability": 1.0,
                "epistemic_quality": 0.8,
                "viewer_relevance": 1.0,
                "specificity": 0.8,
                "needs_verification": False,
                "timestamp_seconds": None,
            }
            for claim in concepts
        ],
    }


def test_selection_prefers_complement_over_duplicate(tmp_path: Path):
    ws = Workspace(tmp_path).ensure()
    videos = [
        {"id": "aaaaaaaaaaa", "title": "A", "url": "https://youtu.be/aaaaaaaaaaa", "duration_seconds": 600, "playlist_index": 1},
        {"id": "bbbbbbbbbbb", "title": "B", "url": "https://youtu.be/bbbbbbbbbbb", "duration_seconds": 600, "playlist_index": 2},
        {"id": "ccccccccccc", "title": "C", "url": "https://youtu.be/ccccccccccc", "duration_seconds": 600, "playlist_index": 3},
    ]
    ws.save_manifest(videos)
    ws.save_analysis("aaaaaaaaaaa", _analysis("aaaaaaaaaaa", ["Use retrieval practice after reading material."]))
    ws.save_analysis("bbbbbbbbbbb", _analysis("bbbbbbbbbbb", ["After reading, practice retrieving the material."]))
    ws.save_analysis("ccccccccccc", _analysis("ccccccccccc", ["Define one concrete next action for every goal."]))
    result = cluster_and_select(
        ws,
        target_coverage=0.95,
        max_minutes=30,
        max_fraction=1.0,
        similarity_threshold=0.45,
        embedding_mode="local",
    )
    picked = [item["video_id"] for item in result["selected"]]
    assert "ccccccccccc" in picked[:2]
    assert not ({"aaaaaaaaaaa", "bbbbbbbbbbb"} <= set(picked[:2]))


def test_single_concept_corpus_does_not_crash(tmp_path: Path):
    ws = Workspace(tmp_path).ensure()
    ws.save_manifest(
        [
            {
                "id": "ddddddddddd",
                "title": "Only video",
                "url": "https://youtu.be/ddddddddddd",
                "duration_seconds": 300,
                "playlist_index": 1,
            }
        ]
    )
    ws.save_analysis(
        "ddddddddddd",
        _analysis("ddddddddddd", ["Write one concrete output after every learning session."]),
    )
    result = cluster_and_select(
        ws,
        target_coverage=0.8,
        max_minutes=10,
        max_fraction=1.0,
        embedding_mode="local",
    )
    assert [item["video_id"] for item in result["selected"]] == ["ddddddddddd"]
    assert result["stop_reason"] == "target_coverage_reached"
