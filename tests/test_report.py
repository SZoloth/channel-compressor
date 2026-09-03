from pathlib import Path

from channel_compressor.report import generate_report
from channel_compressor.utils import atomic_write_json
from channel_compressor.workspace import Workspace


def test_report_flags_partial_corpus_and_renders_html(tmp_path: Path):
    workspace = Workspace(tmp_path).ensure()
    workspace.save_manifest(
        [
            {
                "id": "aaaaaaaaaaa",
                "title": "Attention | retrieval",
                "url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                "duration_seconds": 600,
                "playlist_index": 1,
            },
            {
                "id": "bbbbbbbbbbb",
                "title": "Missing transcript",
                "url": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
                "duration_seconds": 600,
                "playlist_index": 2,
            },
        ]
    )
    workspace.save_run_state({"channel_title": "Mock Channel"})
    workspace.save_transcript(
        "aaaaaaaaaaa",
        {
            "source": "youtube-transcript-api",
            "text": "Retrieve ideas after reading them.",
            "segments": [{"text": "Retrieve ideas after reading them.", "start": 12, "duration": 3}],
        },
    )
    workspace.save_analysis(
        "aaaaaaaaaaa",
        {
            "summary": "A retrieval practice protocol.",
            "concepts": [
                {
                    "claim": "Retrieve ideas after reading them.",
                    "salience": 1,
                    "timestamp_seconds": 12,
                }
            ],
            "cautions": [],
        },
    )
    atomic_write_json(
        workspace.selection_path,
        {
            "eligible_video_count": 1,
            "target_coverage": 0.8,
            "achieved_coverage": 0.9,
            "selected_minutes": 10,
            "stop_reason": "target_coverage_reached",
            "selected": [
                {
                    "rank": 1,
                    "video_id": "aaaaaaaaaaa",
                    "minutes": 10,
                    "marginal_share": 0.9,
                    "cumulative_coverage": 0.9,
                    "novelty_share": 1.0,
                    "consume_mode": "watch key sections",
                    "unique_cluster_labels": ["retrieval | feedback"],
                }
            ],
            "leftovers": [],
        },
    )
    atomic_write_json(
        workspace.clusters_path,
        {
            "clusters": [
                {
                    "cluster_id": "c1",
                    "label": "retrieval | feedback",
                    "importance": 1.0,
                    "video_count": 1,
                    "member_count": 1,
                    "needs_verification_share": 0.0,
                    "example_claims": ["Retrieve ideas | inspect errors"],
                }
            ]
        },
    )

    markdown_path = generate_report(workspace)
    markdown = markdown_path.read_text(encoding="utf-8")
    html = (workspace.outputs_dir / "report.html").read_text(encoding="utf-8")

    assert "**Provisional result.**" in markdown
    assert "analyzed subset" in markdown
    assert "Attention \\| retrieval" in markdown
    assert "<h1>" in html
    assert "<table>" in html
    assert "<pre># Mock Channel" not in html
    assert (workspace.outputs_dir / "ranked_videos.csv").exists()
