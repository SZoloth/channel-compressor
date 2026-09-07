"""Read-only, transcript-free audit. Run: python scripts/audit_corpus.py WORKSPACE."""
import hashlib
import json
import sys
from pathlib import Path


def audit(root: Path) -> dict:
    videos = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    ids = [v["id"] for v in videos]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("duplicate manifest IDs")
    hashes = []
    words = analyzed = 0
    missing = []
    for video_id in ids:
        path = root / "transcripts" / f"{video_id}.json"
        transcript = json.loads(path.read_text()) if path.exists() else {}
        text = transcript.get("text", "")
        if not text.strip():
            missing.append(video_id)
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        hashes.append(digest)
        words += len(text.split())
        if transcript.get("text_sha256") != digest:
            errors.append(f"transcript hash mismatch: {video_id}")
        path = root / "analyses" / f"{video_id}.json"
        analysis = json.loads(path.read_text()) if path.exists() else {}
        if not analysis.get("concepts") or not analysis.get("summary"):
            errors.append(f"missing or empty analysis: {video_id}")
        elif analysis.get("transcript_sha256") != digest:
            errors.append(f"stale analysis: {video_id}")
        else:
            analyzed += 1
    if len(hashes) != len(set(hashes)):
        errors.append("identical transcript hashes across videos")
    return {
        "discovered": len(ids), "transcripts": len(hashes), "analyzed": analyzed,
        "words": words, "source_minutes": sum(v.get("duration_seconds", 0) or 0 for v in videos) / 60,
        "missing_transcript_ids": sorted(missing), "integrity_errors": errors,
        "complete": bool(ids) and not missing and not errors,
    }


if __name__ == "__main__":
    result = audit(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["integrity_errors"] else 0)
