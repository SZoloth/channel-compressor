import importlib.util
import json
from pathlib import Path

from channel_compressor.utils import sha256_text

spec = importlib.util.spec_from_file_location(
    "corpus_audit", Path(__file__).parents[1] / "scripts" / "audit_corpus.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_missing_and_stale(tmp_path):
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"id": "test", "duration_seconds": 60}))
    result = module.audit(tmp_path)
    assert result["complete"] is False
    assert result["missing_transcript_ids"] == ["test"]
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "analyses").mkdir()
    text = "A nonempty transcript for a deterministic audit."
    digest = sha256_text(text)
    (tmp_path / "transcripts/test.json").write_text(json.dumps({"text": text, "text_sha256": digest}))
    analysis = {"summary": "Summary", "concepts": ["idea"], "transcript_sha256": "stale"}
    (tmp_path / "analyses/test.json").write_text(json.dumps(analysis))
    assert module.audit(tmp_path)["integrity_errors"] == ["stale analysis: test"]
    analysis["transcript_sha256"] = digest
    (tmp_path / "analyses/test.json").write_text(json.dumps(analysis))
    assert module.audit(tmp_path)["complete"] is True
