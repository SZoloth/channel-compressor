from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .utils import atomic_write_text, load_json, seconds_to_timestamp
from .workspace import Workspace


def _youtube_timestamp_url(url: str, seconds: float | None) -> str:
    if seconds is None:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode({'t': f'{int(seconds)}s'})}"


def _fmt_minutes(value: float) -> str:
    minutes = int(round(value))
    hours, rest = divmod(minutes, 60)
    return f"{hours}h {rest}m" if hours else f"{rest}m"


def _md_text(value: Any) -> str:
    """Make untrusted/generated text safe inside a Markdown table cell."""
    text = " ".join(str(value or "").split())
    return text.replace("\\", "\\\\").replace("|", "\\|")


def _md_link(label: Any, url: Any) -> str:
    safe_label = _md_text(label).replace("[", "\\[").replace("]", "\\]")
    safe_url = str(url or "").replace(" ", "%20").replace(")", "%29")
    return f"[{safe_label}]({safe_url})" if safe_url else safe_label


def _stop_reason(value: str | None) -> str:
    return {
        "target_coverage_reached": "coverage target reached",
        "max_fraction_reached": "maximum channel fraction reached",
        "time_budget_reached": "time budget reached",
        "no_additional_value": "no remaining video added material value",
    }.get(str(value or ""), str(value or "not recorded").replace("_", " "))


def generate_report(workspace: Workspace, *, title: str | None = None) -> Path:
    manifest = workspace.manifest_by_id()
    run_state = workspace.load_run_state()
    selection = load_json(workspace.selection_path, default={}) or {}
    cluster_payload = load_json(workspace.clusters_path, default={}) or {}
    clusters = cluster_payload.get("clusters") or []
    selected = selection.get("selected") or []
    leftovers = selection.get("leftovers") or []

    transcript_payloads = {
        video_id: workspace.load_transcript(video_id) for video_id in manifest
    }
    transcript_count = sum(bool(item and item.get("text")) for item in transcript_payloads.values())
    transcript_sources = Counter(
        str(item.get("source") or "unknown")
        for item in transcript_payloads.values()
        if item and item.get("text")
    )
    analyses = {video_id: workspace.load_analysis(video_id) for video_id in manifest}
    analysis_count = sum(bool(item) for item in analyses.values())
    complete = bool(manifest) and transcript_count == len(manifest) and analysis_count == len(manifest)
    eligible_count = int(selection.get("eligible_video_count") or analysis_count)
    report_title = title or (
        f"{run_state.get('channel_title') or 'YouTube channel'} — 80/20 consumption map"
    )

    lines = [
        f"# {report_title}",
        "",
        "> The selection optimizes **weighted marginal concept coverage**, not views. "
        "Repeated ideas can establish a theme as central, but a second video receives little "
        "credit when the chosen set already covers it.",
        "",
    ]
    if not complete:
        lines.extend(
            [
                "> [!WARNING]",
                f"> **Provisional result.** The pipeline captured {transcript_count}/{len(manifest)} "
                f"transcripts and analyzed {analysis_count}/{len(manifest)} discovered videos. "
                "The coverage percentage below describes the analyzed subset—not unseen videos.",
                "",
            ]
        )

    source_summary = ", ".join(
        f"{source}: {count}" for source, count in transcript_sources.most_common()
    ) or "none"
    lines.extend(
        [
            "## Corpus",
            "",
            f"- Videos discovered: **{len(manifest)}**",
            f"- Transcripts captured: **{transcript_count}** ({source_summary})",
            f"- Videos analyzed and eligible for selection: **{eligible_count}**",
            f"- Selection: **{len(selected)} videos / {_fmt_minutes(float(selection.get('selected_minutes', 0)))}**",
            f"- Weighted concept coverage{' of analyzed subset' if not complete else ''}: "
            f"**{float(selection.get('achieved_coverage', 0)):.1%}** "
            f"(target {float(selection.get('target_coverage', 0.8)):.0%})",
            f"- Selector stopped because: **{_stop_reason(selection.get('stop_reason'))}**",
            "",
            "## Consume these",
            "",
            "| # | Video | Mode | Time | New value | Cumulative | Why it survives de-duplication |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for item in selected:
        video_id = str(item["video_id"])
        video = manifest.get(video_id, {})
        labels = "; ".join(item.get("unique_cluster_labels") or [])
        lines.append(
            f"| {item.get('rank')} | {_md_link(video.get('title', video_id), video.get('url'))} | "
            f"{_md_text(item.get('consume_mode'))} | {_fmt_minutes(float(item.get('minutes', 0)))} | "
            f"{float(item.get('marginal_share', 0)):.1%} | "
            f"{float(item.get('cumulative_coverage', 0)):.1%} | {_md_text(labels)} |"
        )
    if not selected:
        lines.append("| — | No selection generated | — | — | — | — | Run analysis first. |")

    lines.extend(["", "## What each selected video contributes", ""])
    for item in selected:
        video_id = str(item["video_id"])
        video = manifest.get(video_id, {})
        analysis = analyses.get(video_id) or {}
        lines.extend(
            [
                f"### {item.get('rank')}. {_md_link(video.get('title', video_id), video.get('url'))}",
                "",
                str(analysis.get("summary") or "No summary available."),
                "",
                f"**Consumption mode:** {item.get('consume_mode')}. "
                f"**Novelty within this video:** {float(item.get('novelty_share', 0)):.0%}.",
                "",
            ]
        )
        concepts = sorted(
            analysis.get("concepts") or [],
            key=lambda concept: float(concept.get("salience", 0)),
            reverse=True,
        )[:4]
        for concept in concepts:
            timestamp = concept.get("timestamp_seconds")
            claim = " ".join(str(concept.get("claim") or "").split())
            if timestamp is not None:
                link = _youtube_timestamp_url(str(video.get("url")), float(timestamp))
                lines.append(f"- [{seconds_to_timestamp(float(timestamp))}]({link}): {claim}")
            else:
                lines.append(f"- {claim}")
        cautions = analysis.get("cautions") or []
        if cautions:
            lines.append(f"- **Caution:** {' '.join(str(cautions[0]).split())}")
        lines.append("")

    lines.extend(
        [
            "## Core ideas across the corpus",
            "",
            "| Theme | Importance | Videos repeating it | Verification risk | Representative paraphrases |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for cluster in clusters[:15]:
        examples = " · ".join(str(item) for item in (cluster.get("example_claims") or [])[:2])
        lines.append(
            f"| {_md_text(cluster.get('label'))} | {float(cluster.get('importance', 0)):.2f} | "
            f"{cluster.get('video_count')} | "
            f"{float(cluster.get('needs_verification_share', 0)):.0%} | {_md_text(examples)} |"
        )
    if not clusters:
        lines.append("| — | — | — | — | No clusters generated. |")

    selected_titles = {
        video_id: manifest.get(video_id, {}).get("title", video_id) for video_id in manifest
    }
    lines.extend(
        [
            "",
            "## Highest-value omissions",
            "",
            "These are not necessarily bad. They are the best remaining candidates after the "
            "selected set, usually because most of their useful content is already represented.",
            "",
            "| Video | Time | Covered by chosen set | Closest chosen video | Overlap with closest |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for item in leftovers[:15]:
        video_id = str(item["video_id"])
        video = manifest.get(video_id, {})
        duplicate_id = item.get("most_redundant_with")
        duplicate_title = selected_titles.get(str(duplicate_id), "—") if duplicate_id else "—"
        lines.append(
            f"| {_md_link(video.get('title', video_id), video.get('url'))} | "
            f"{_fmt_minutes(float(item.get('minutes', 0)))} | "
            f"{float(item.get('covered_by_selected_share', 0)):.0%} | "
            f"{_md_text(duplicate_title)} | "
            f"{float(item.get('closest_selected_overlap_share', 0)):.0%} |"
        )
    if not leftovers:
        lines.append("| — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## How to use this report",
            "",
            "1. Start with items marked **watch**; their demonstration, delivery, or narrative "
            "carries value beyond the text.",
            "2. For **read transcript/summary**, read the synthesis first and open the source only "
            "where the idea is directly useful.",
            "3. Treat neuroscience mechanisms flagged for verification as hypotheses. A tactic can "
            "be worth testing even when its explanatory wrapper is too tidy.",
            "4. Re-run the tool later. Discovery and processing are cached, so only new or changed "
            "videos need work.",
            "",
            "## Interpretation boundary",
            "",
            "“80%” is a model-based estimate of weighted, viewer-specific concept value. It is not "
            "a claim that ideas have an objective cardinal value, nor that watching replaces doing. "
            "The report is most useful as a queue-pruning instrument; the selected ideas still need "
            "to be tested through output, feedback, and repetition.",
            "",
        ]
    )

    output = workspace.outputs_dir / "report.md"
    atomic_write_text(output, "\n".join(lines))
    _write_csv_outputs(workspace, manifest, selected, leftovers, clusters)
    _write_html(output, workspace.outputs_dir / "report.html", report_title)
    return output


def _write_csv_outputs(
    workspace: Workspace,
    manifest: dict[str, dict[str, Any]],
    selected: list[dict[str, Any]],
    leftovers: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
) -> None:
    selected_by_id = {str(item["video_id"]): item for item in selected}
    leftover_by_id = {str(item["video_id"]): item for item in leftovers}
    ranked_path = workspace.outputs_dir / "ranked_videos.csv"
    with ranked_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "video_id",
            "title",
            "url",
            "selected_rank",
            "consume_mode",
            "duration_minutes",
            "marginal_share",
            "cumulative_coverage",
            "novelty_share",
            "covered_by_selected_share",
            "closest_selected_overlap_share",
            "most_redundant_with",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for video_id, video in manifest.items():
            chosen = selected_by_id.get(video_id, {})
            leftover = leftover_by_id.get(video_id, {})
            writer.writerow(
                {
                    "video_id": video_id,
                    "title": video.get("title"),
                    "url": video.get("url"),
                    "selected_rank": chosen.get("rank"),
                    "consume_mode": chosen.get("consume_mode"),
                    "duration_minutes": chosen.get("minutes") or leftover.get("minutes"),
                    "marginal_share": chosen.get("marginal_share"),
                    "cumulative_coverage": chosen.get("cumulative_coverage"),
                    "novelty_share": chosen.get("novelty_share"),
                    "covered_by_selected_share": leftover.get("covered_by_selected_share"),
                    "closest_selected_overlap_share": leftover.get(
                        "closest_selected_overlap_share"
                    ),
                    "most_redundant_with": leftover.get("most_redundant_with"),
                }
            )

    clusters_path = workspace.outputs_dir / "concept_clusters.csv"
    with clusters_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "cluster_id",
            "label",
            "importance",
            "video_count",
            "member_count",
            "needs_verification_share",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cluster in clusters:
            writer.writerow({field: cluster.get(field) for field in fields})

    readwise_path = workspace.outputs_dir / "readwise_queue.csv"
    with readwise_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "rank",
            "title",
            "url",
            "consume_mode",
            "marginal_share",
            "cumulative_coverage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            video = manifest.get(str(item["video_id"]), {})
            writer.writerow(
                {
                    "rank": item.get("rank"),
                    "title": video.get("title"),
                    "url": video.get("url"),
                    "consume_mode": item.get("consume_mode"),
                    "marginal_share": item.get("marginal_share"),
                    "cumulative_coverage": item.get("cumulative_coverage"),
                }
            )


def _write_html(markdown_path: Path, html_path: Path, title: str) -> None:
    markdown_text = markdown_path.read_text(encoding="utf-8")
    try:
        import mistune

        renderer = mistune.create_markdown(escape=True, plugins=["table", "strikethrough"])
        body = renderer(markdown_text)
    except (ImportError, RuntimeError, ValueError):
        body = f"<pre>{html.escape(markdown_text)}</pre>"

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#171717; --muted:#686868; --line:#dedbd3; --paper:#fbfaf7; --card:#fff; --accent:#1f4b3f; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.62 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1120px; margin:0 auto; padding:56px 28px 112px; }}
h1 {{ font-size:clamp(2rem,5vw,4.2rem); line-height:1.02; letter-spacing:-.045em; max-width:980px; margin:0 0 32px; }}
h2 {{ margin:56px 0 18px; font-size:1.6rem; letter-spacing:-.02em; }}
h3 {{ margin:34px 0 10px; font-size:1.18rem; }}
p,li {{ max-width:78ch; }}
a {{ color:var(--accent); text-decoration-thickness:1px; text-underline-offset:3px; }}
blockquote {{ margin:24px 0; padding:16px 20px; border-left:4px solid var(--accent); background:#f1f4f0; color:#2f3c38; }}
blockquote p {{ margin:0; }}
table {{ width:100%; border-collapse:separate; border-spacing:0; margin:20px 0 36px; font-size:.91rem; background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
th,td {{ padding:12px 14px; vertical-align:top; text-align:left; border-bottom:1px solid var(--line); }}
th {{ background:#f1efe9; font-weight:650; }}
tr:last-child td {{ border-bottom:0; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; padding:18px; background:#f1efe9; border-radius:10px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9em; }}
hr {{ border:0; border-top:1px solid var(--line); margin:48px 0; }}
@media (max-width:760px) {{ main {{ padding:34px 16px 72px; }} table {{ display:block; overflow-x:auto; }} th,td {{ min-width:110px; }} }}
@media (prefers-color-scheme:dark) {{
  :root {{ color-scheme:dark; --ink:#f1eee7; --muted:#aaa59c; --line:#45423d; --paper:#151513; --card:#1d1d1a; --accent:#9fd1bd; }}
  blockquote {{ background:#202923; color:#d7e6dd; }} th,pre {{ background:#272621; }}
}}
</style>
</head>
<body><main>{body}</main></body>
</html>"""
    atomic_write_text(html_path, document)
