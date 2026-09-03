from __future__ import annotations

import json
import os
from typing import Any

from .profile import profile_as_prompt
from .utils import chunk_text, clamp, extract_json_object

VIDEO_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["principle", "method", "evidence", "example", "biography"],
                    },
                    "salience": {"type": "number"},
                    "actionability": {"type": "number"},
                    "epistemic_quality": {"type": "number"},
                    "viewer_relevance": {"type": "number"},
                    "specificity": {"type": "number"},
                    "needs_verification": {"type": "boolean"},
                    "timestamp_seconds": {"type": ["number", "null"]},
                },
                "required": [
                    "claim",
                    "kind",
                    "salience",
                    "actionability",
                    "epistemic_quality",
                    "viewer_relevance",
                    "specificity",
                    "needs_verification",
                    "timestamp_seconds",
                ],
                "additionalProperties": False,
            },
        },
        "watchability": {"type": "number"},
        "compressibility": {"type": "number"},
        "audience_fit": {"type": "number"},
        "within_video_repetition": {"type": "number"},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "summary",
        "concepts",
        "watchability",
        "compressibility",
        "audience_fit",
        "within_video_repetition",
        "cautions",
    ],
    "additionalProperties": False,
}


class OpenAIAnalyzer:
    def __init__(self, model: str | None = None) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The OpenAI SDK is not installed. Install with `pip install -e '.[openai]'`."
            ) from exc
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-5.6-luna"

    def _response_json(self, developer: str, user: str) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            store=False,
            max_output_tokens=4_000,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "channel_compressor_video_analysis",
                    "schema": VIDEO_ANALYSIS_SCHEMA,
                    "strict": True,
                }
            },
            input=[
                {"role": "developer", "content": developer},
                {"role": "user", "content": user},
            ],
        )
        if not response.output_text:
            raise RuntimeError("The model returned no structured analysis")
        return extract_json_object(response.output_text)

    def analyze_video(
        self,
        *,
        title: str,
        transcript: str,
        profile: dict[str, Any],
        max_concepts: int = 9,
    ) -> dict[str, Any]:
        chunks = chunk_text(transcript, max_chars=14_000, overlap_chars=600)
        partials: list[dict[str, Any]] = []
        developer = """You are an exacting corpus analyst. Extract the durable informational value of a video transcript, not its rhetoric. Treat neuroscience explanations as claims, not facts. Distinguish practical usefulness from evidentiary strength. Return only valid JSON; no Markdown."""
        schema = """Return this object:
{
  "summary": "2-4 sentence neutral summary",
  "concepts": [
    {
      "claim": "one atomic paraphrased idea; no quotation",
      "kind": "principle|method|evidence|example|biography",
      "salience": 0.0,
      "actionability": 0.0,
      "epistemic_quality": 0.0,
      "viewer_relevance": 0.0,
      "specificity": 0.0,
      "needs_verification": false,
      "timestamp_seconds": null
    }
  ],
  "watchability": 0.0,
  "compressibility": 0.0,
  "audience_fit": 0.0,
  "within_video_repetition": 0.0,
  "cautions": ["short paraphrased caution"]
}
All scores are 0-1. `epistemic_quality` evaluates support PRESENTED in the transcript, not whether the claim happens to be true. `compressibility` means how fully a written synthesis can replace watching. Keep only substantive concepts. For `timestamp_seconds`, use the nearest explicit [m:ss] or [h:mm:ss] marker before the idea; otherwise return null."""
        profile_text = profile_as_prompt(profile)
        for index, chunk in enumerate(chunks, start=1):
            partials.append(
                self._response_json(
                    developer,
                    f"Video title: {title}\nChunk {index}/{len(chunks)}\n\nViewer profile:\n{profile_text}\n\n{schema}\n\nTranscript chunk:\n{chunk}",
                )
            )
        if len(partials) == 1:
            result = partials[0]
        else:
            result = self._response_json(
                developer,
                f"Consolidate the following chunk analyses for one video titled {title!r}. "
                f"Merge semantic duplicates and keep at most {max_concepts} concepts. "
                f"Use the same schema. Viewer profile:\n{profile_text}\n\nChunk analyses:\n{json.dumps(partials, ensure_ascii=False)}",
            )
        result["concepts"] = _sanitize_concepts(result.get("concepts") or [], max_concepts)
        for field in (
            "watchability",
            "compressibility",
            "audience_fit",
            "within_video_repetition",
        ):
            result[field] = clamp(result.get(field, 0.5))
        result["model"] = self.model
        result["mode"] = "openai"
        return result

    def embed(self, texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):
            batch = texts[start : start + 128]
            response = self.client.embeddings.create(model=model, input=batch)
            vectors.extend([list(item.embedding) for item in response.data])
        return vectors


def _sanitize_concepts(raw: list[Any], maximum: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    allowed_kinds = {"principle", "method", "evidence", "example", "biography"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        if len(claim.split()) < 4:
            continue
        kind = str(item.get("kind") or "principle").lower()
        if kind not in allowed_kinds:
            kind = "principle"
        timestamp = item.get("timestamp_seconds")
        try:
            timestamp = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp = None
        output.append(
            {
                "claim": claim,
                "kind": kind,
                "salience": clamp(item.get("salience", 0.5)),
                "actionability": clamp(item.get("actionability", 0.5)),
                "epistemic_quality": clamp(item.get("epistemic_quality", 0.5)),
                "viewer_relevance": clamp(item.get("viewer_relevance", 0.5)),
                "specificity": clamp(item.get("specificity", 0.5)),
                "needs_verification": bool(item.get("needs_verification", False)),
                "timestamp_seconds": timestamp,
            }
        )
        if len(output) >= maximum:
            break
    return output
