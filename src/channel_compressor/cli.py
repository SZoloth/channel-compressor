from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from .analysis import analyze_transcripts, cluster_and_select
from .discovery import discover_channel
from .imports import import_transcripts
from .readwise import (
    fetch_reader_transcripts,
    import_manifest_to_reader,
    sync_selection_to_reader,
)
from .report import generate_report
from .transcripts import transcribe_workspace
from .utils import load_json
from .workspace import Workspace

app = typer.Typer(
    no_args_is_help=True,
    help="Capture a YouTube channel, de-duplicate its ideas, and find the smallest high-value subset.",
)
console = Console()
DEFAULT_CHANNEL = "https://www.youtube.com/@erinmerylstudy/videos"
DEFAULT_WORKSPACE = Path("channel-compressor-workspace")


def _workspace(path: Path) -> Workspace:
    return Workspace(path.expanduser().resolve()).ensure()


def _progress_callback(label: str):
    progress = Progress()
    task = progress.add_task(label, total=None)
    progress.start()

    def callback(index: int, total: int, video_id: str, status: str) -> None:
        progress.update(task, total=total, completed=index, description=f"{label}: {status} ({video_id})")
        if index >= total:
            progress.stop()

    return callback, progress


@app.command()
def discover(
    channel_url: str = typer.Argument(DEFAULT_CHANNEL),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    limit: int | None = typer.Option(None, min=1),
    include_lives: bool = typer.Option(False),
    include_shorts: bool = typer.Option(False),
) -> None:
    """Inventory all videos in a channel tab."""
    ws = _workspace(workspace)
    videos = discover_channel(
        channel_url,
        ws,
        limit=limit,
        include_lives=include_lives,
        include_shorts=include_shorts,
    )
    console.print(f"Discovered [bold]{len(videos)}[/bold] videos → {ws.manifest_path}")


@app.command()
def transcribe(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    providers: str = typer.Option("youtube,ytdlp,whisper"),
    languages: str = typer.Option("en,en-US,en-GB"),
    force: bool = typer.Option(False),
    limit: int | None = typer.Option(None, min=1),
    delay_seconds: float = typer.Option(0.4, min=0.0),
    request_timeout_seconds: float = typer.Option(30.0, min=1.0),
    whisper_model: str = typer.Option("small.en"),
    whisper_device: str = typer.Option("cpu"),
    whisper_compute_type: str = typer.Option("int8"),
) -> None:
    """Capture transcripts with an explicit fallback chain."""
    ws = _workspace(workspace)
    callback, progress = _progress_callback("Transcribing")
    try:
        counts = transcribe_workspace(
            ws,
            providers=[item.strip() for item in providers.split(",") if item.strip()],
            languages=[item.strip() for item in languages.split(",") if item.strip()],
            force=force,
            limit=limit,
            delay_seconds=delay_seconds,
            request_timeout_seconds=request_timeout_seconds,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            on_progress=callback,
        )
    finally:
        progress.stop()
    console.print(counts)


@app.command(name="transcript-import")
def transcript_import(
    source: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="A JSON/JSONL/TXT/VTT/SRT file or directory exported by another transcript tool.",
    ),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    source_label: str = typer.Option("external", help="Short provenance label stored in cache."),
    overwrite: bool = typer.Option(False),
    minimum_words: int = typer.Option(40, min=1),
) -> None:
    """Import transcript exports for videos already in the manifest."""
    counts = import_transcripts(
        _workspace(workspace),
        source,
        source_label=source_label,
        overwrite=overwrite,
        minimum_words=minimum_words,
    )
    console.print(counts)


@app.command()
def analyze(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    profile: Path | None = typer.Option(None, "--profile", "-p"),
    mode: str = typer.Option("auto", help="auto, openai, or local"),
    model: str | None = typer.Option(None, help="OpenAI model; otherwise OPENAI_MODEL/default"),
    force: bool = typer.Option(False),
    limit: int | None = typer.Option(None, min=1),
    target_coverage: float = typer.Option(0.8, min=0.05, max=1.0),
    max_minutes: float | None = typer.Option(120.0, min=1.0),
    max_fraction: float = typer.Option(0.2, min=0.01, max=1.0),
    similarity_threshold: float = typer.Option(0.76, min=0.3, max=0.99),
) -> None:
    """Extract ideas, cluster repetitions, and run weighted set cover."""
    ws = _workspace(workspace)
    callback, progress = _progress_callback("Analyzing")
    try:
        counts = analyze_transcripts(
            ws,
            profile_path=profile,
            mode=mode,
            model=model,
            force=force,
            limit=limit,
            on_progress=callback,
        )
    finally:
        progress.stop()
    selection = cluster_and_select(
        ws,
        target_coverage=target_coverage,
        max_minutes=max_minutes,
        max_fraction=max_fraction,
        similarity_threshold=similarity_threshold,
        embedding_mode="openai" if mode == "openai" else "auto" if mode == "auto" else "local",
        model=model,
    )
    report_path = generate_report(ws)
    console.print(counts)
    console.print(
        f"Selected [bold]{len(selection['selected'])}[/bold] videos, "
        f"covering [bold]{selection['achieved_coverage']:.1%}[/bold] of weighted value → {report_path}"
    )


@app.command(name="report")
def report_command(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    title: str | None = typer.Option(None),
) -> None:
    """Regenerate Markdown, HTML, and CSV outputs from cached analysis."""
    path = generate_report(_workspace(workspace), title=title)
    console.print(f"Report → {path}")


@app.command(name="reader-import")
def reader_import(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    tag: str = typer.Option("channel-compressor-corpus"),
    location: str = typer.Option("archive"),
    force: bool = typer.Option(False),
) -> None:
    """Optionally submit every URL to Reader, tagged and archived."""
    ws = _workspace(workspace)
    callback, progress = _progress_callback("Saving to Reader")
    try:
        counts = import_manifest_to_reader(
            ws, tag=tag, location=location, force=force, on_progress=lambda i, n, s: callback(i, n, "reader", s)
        )
    finally:
        progress.stop()
    console.print(counts)


@app.command(name="reader-fetch")
def reader_fetch(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    tag: str = typer.Option("channel-compressor-corpus"),
    overwrite: bool = typer.Option(False),
) -> None:
    """Fetch transcript HTML Reader has generated into the local provider cache."""
    counts = fetch_reader_transcripts(_workspace(workspace), tag=tag, overwrite=overwrite)
    console.print(counts)


@app.command(name="reader-sync")
def reader_sync(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    tag: str = typer.Option("channel-compressor-watch"),
    location: str = typer.Option("later"),
) -> None:
    """Save only the final 80/20 selection to Reader with summaries."""
    counts = sync_selection_to_reader(_workspace(workspace), tag=tag, location=location)
    console.print(counts)


@app.command()
def status(
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
) -> None:
    """Show resumable pipeline state."""
    ws = _workspace(workspace)
    manifest = ws.load_manifest()
    transcripts = sum(ws.transcript_path(str(item["id"])).exists() for item in manifest)
    analyses = sum(ws.analysis_path(str(item["id"])).exists() for item in manifest)
    selection = load_json(ws.selection_path, default={}) or {}
    console.print(
        {
            "workspace": str(ws.root),
            "videos": len(manifest),
            "transcripts": transcripts,
            "analyses": analyses,
            "selected": len(selection.get("selected") or []),
            "coverage": selection.get("achieved_coverage"),
        }
    )


@app.command(name="run")
def run_pipeline(
    channel_url: str = typer.Argument(DEFAULT_CHANNEL),
    workspace: Path = typer.Option(DEFAULT_WORKSPACE, "--workspace", "-w"),
    profile: Path | None = typer.Option(None, "--profile", "-p"),
    providers: str = typer.Option("youtube,ytdlp,whisper"),
    analysis_mode: str = typer.Option("auto", help="auto, openai, or local"),
    model: str | None = typer.Option(None),
    target_coverage: float = typer.Option(0.8, min=0.05, max=1.0),
    max_minutes: float | None = typer.Option(120.0, min=1.0),
    max_fraction: float = typer.Option(0.2, min=0.01, max=1.0),
    similarity_threshold: float = typer.Option(0.76, min=0.3, max=0.99),
    limit: int | None = typer.Option(None, min=1),
    force: bool = typer.Option(False),
    request_timeout_seconds: float = typer.Option(30.0, min=1.0),
) -> None:
    """Run discovery → transcript capture → analysis → report."""
    ws = _workspace(workspace)
    console.rule("Discovery")
    videos = discover_channel(channel_url, ws, limit=limit)
    console.print(f"Discovered {len(videos)} videos")

    console.rule("Transcripts")
    callback, progress = _progress_callback("Transcribing")
    try:
        transcript_counts = transcribe_workspace(
            ws,
            providers=[item.strip() for item in providers.split(",") if item.strip()],
            limit=limit,
            force=force,
            request_timeout_seconds=request_timeout_seconds,
            on_progress=callback,
        )
    finally:
        progress.stop()
    console.print(transcript_counts)

    console.rule("Analysis")
    callback, progress = _progress_callback("Analyzing")
    try:
        analysis_counts = analyze_transcripts(
            ws,
            profile_path=profile,
            mode=analysis_mode,
            model=model,
            force=force,
            limit=limit,
            on_progress=callback,
        )
    finally:
        progress.stop()
    console.print(analysis_counts)

    selection = cluster_and_select(
        ws,
        target_coverage=target_coverage,
        max_minutes=max_minutes,
        max_fraction=max_fraction,
        similarity_threshold=similarity_threshold,
        embedding_mode="openai" if analysis_mode == "openai" else "local" if analysis_mode == "local" else "auto",
        model=model,
    )
    report_path = generate_report(ws)
    console.rule("Done")
    console.print(
        f"[bold]{len(selection['selected'])}[/bold] videos cover "
        f"[bold]{selection['achieved_coverage']:.1%}[/bold] of weighted concept value."
    )
    console.print(f"Markdown: {report_path}")
    console.print(f"HTML: {ws.outputs_dir / 'report.html'}")
    console.print(f"CSV: {ws.outputs_dir / 'ranked_videos.csv'}")


if __name__ == "__main__":
    app()
