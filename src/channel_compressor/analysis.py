from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .llm import OpenAIAnalyzer
from .profile import load_profile, profile_keywords
from .utils import (
    atomic_write_json,
    clamp,
    normalize_for_match,
    safe_float,
    sha256_text,
    split_sentences,
    strip_cta_sentences,
    timestamped_text,
    write_jsonl,
)
from .workspace import Workspace

KIND_WEIGHT = {
    "principle": 1.0,
    "method": 1.12,
    "evidence": 0.72,
    "example": 0.32,
    "biography": 0.48,
}
NEURO_CLAIM_RE = re.compile(
    r"\b(?:dopamine|neuroplastic|neuron|neural|prefrontal|working memory|brain-derived|bdnf|cortisol|synaptic|mesolimbic)\b",
    re.IGNORECASE,
)
METHOD_RE = re.compile(
    r"\b(?:do|write|read|practice|set|make|build|remove|schedule|record|explain|review|repeat|choose|start|stop|use)\b",
    re.IGNORECASE,
)


def _local_relevance(text: str, profile: dict[str, Any]) -> float:
    positive, negative = profile_keywords(profile)
    words = set(normalize_for_match(text).split())
    positive_score = sum(weight for word, weight in positive.items() if word in words)
    negative_score = sum(weight for word, weight in negative.items() if word in words)
    return clamp(0.48 + 0.08 * positive_score - 0.1 * negative_score)


def _mmr_sentences(sentences: list[str], maximum: int = 9) -> list[tuple[str, float]]:
    candidates = [
        sentence
        for sentence in sentences
        if 8 <= len(sentence.split()) <= 70 and not sentence.lower().startswith(("hey ", "hi "))
    ]
    if not candidates:
        return []
    if len(candidates) == 1:
        return [(candidates[0], 1.0)]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(candidates)
    centroid = np.asarray(matrix.mean(axis=0))
    relevance = cosine_similarity(matrix, centroid).reshape(-1)
    similarity = cosine_similarity(matrix)
    selected: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < maximum:
        best_index = -1
        best_score = -1e9
        for index in remaining:
            redundancy = max((similarity[index, prior] for prior in selected), default=0.0)
            position_bonus = 0.05 * (1.0 - index / max(1, len(candidates) - 1))
            score = 0.76 * relevance[index] - 0.24 * redundancy + position_bonus
            if score > best_score:
                best_score = float(score)
                best_index = index
        selected.append(best_index)
        remaining.remove(best_index)
    max_rel = max(float(relevance[index]) for index in selected) or 1.0
    return [(candidates[index], clamp(float(relevance[index]) / max_rel)) for index in selected]


def local_analyze_video(
    title: str,
    text: str,
    profile: dict[str, Any],
    maximum: int = 9,
) -> dict[str, Any]:
    cleaned = strip_cta_sentences(text)
    sentences = split_sentences(cleaned)
    ranked = _mmr_sentences(sentences, maximum=maximum)
    concepts: list[dict[str, Any]] = []
    for sentence, salience in ranked:
        lower = sentence.lower()
        kind = "method" if METHOD_RE.search(sentence) else "principle"
        actionability = 0.78 if kind == "method" else 0.45
        evidence_terms = sum(
            token in lower
            for token in ("study", "research", "data", "experiment", "meta-analysis", "paper")
        )
        epistemic = clamp(0.42 + 0.12 * evidence_terms)
        needs_verification = bool(NEURO_CLAIM_RE.search(sentence)) and evidence_terms == 0
        if needs_verification:
            epistemic = min(epistemic, 0.38)
        concepts.append(
            {
                "claim": sentence,
                "kind": kind,
                "salience": salience,
                "actionability": actionability,
                "epistemic_quality": epistemic,
                "viewer_relevance": _local_relevance(f"{title} {sentence}", profile),
                "specificity": clamp(0.35 + min(0.45, len(sentence.split()) / 100)),
                "needs_verification": needs_verification,
                "timestamp_seconds": None,
            }
        )
    summary = " ".join(item[0] for item in ranked[:3])
    title_lower = title.lower()
    audience_fit = _local_relevance(title, profile)
    if any(term in title_lower for term in ("exam", "gcse", "a level", "study with me")):
        audience_fit = max(0.1, audience_fit - 0.25)
    watchability = 0.68 if any(term in title_lower for term in ("vlog", "tour", "story", "interview")) else 0.38
    compressibility = 0.82 if watchability < 0.5 else 0.52
    return {
        "summary": summary,
        "concepts": concepts,
        "watchability": watchability,
        "compressibility": compressibility,
        "audience_fit": audience_fit,
        "within_video_repetition": 0.5,
        "cautions": [
            "Contains neuroscience language without evidence in the transcript."
            for item in concepts
            if item["needs_verification"]
        ][:1],
        "mode": "local",
        "model": "tfidf-mmr",
    }


def analyze_transcripts(
    workspace: Workspace,
    *,
    profile_path: Path | None,
    mode: str = "auto",
    model: str | None = None,
    force: bool = False,
    limit: int | None = None,
    on_progress: Callable[[int, int, str, str], None] | None = None,
) -> dict[str, int]:
    profile = load_profile(profile_path)
    profile_hash = sha256_text(
        json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    videos = workspace.load_manifest()
    if limit:
        videos = videos[:limit]
    requested_mode = mode.lower()
    if requested_mode not in {"auto", "openai", "local"}:
        raise ValueError("Analysis mode must be one of: auto, openai, local")
    use_openai = requested_mode == "openai" or (
        requested_mode == "auto" and bool(os.getenv("OPENAI_API_KEY"))
    )
    analyzer = OpenAIAnalyzer(model=model) if use_openai else None
    desired_mode = "openai" if analyzer else "local"
    desired_model = analyzer.model if analyzer else "tfidf-mmr"
    counts = {"analyzed": 0, "cached": 0, "missing_transcript": 0, "failed": 0}

    for index, video in enumerate(videos, start=1):
        video_id = str(video["id"])
        transcript = workspace.load_transcript(video_id)
        if not transcript or not transcript.get("text"):
            counts["missing_transcript"] += 1
            if on_progress:
                on_progress(index, len(videos), video_id, "missing transcript")
            continue
        transcript_hash = transcript.get("text_sha256") or sha256_text(str(transcript["text"]))
        existing = workspace.load_analysis(video_id)
        if (
            existing
            and not force
            and existing.get("transcript_sha256") == transcript_hash
            and existing.get("profile_sha256") == profile_hash
            and existing.get("mode") == desired_mode
            and existing.get("model") == desired_model
        ):
            counts["cached"] += 1
            if on_progress:
                on_progress(index, len(videos), video_id, "cached")
            continue
        try:
            if analyzer:
                input_text = timestamped_text(transcript.get("segments") or [])
                if not input_text:
                    input_text = str(transcript["text"])
                try:
                    analysis = analyzer.analyze_video(
                        title=str(video.get("title") or video_id),
                        transcript=input_text,
                        profile=profile,
                    )
                except Exception as primary_exc:
                    if requested_mode != "auto":
                        raise
                    analysis = local_analyze_video(
                        str(video.get("title") or video_id), str(transcript["text"]), profile
                    )
                    analysis["fallback_from"] = "openai"
                    analysis["fallback_reason"] = type(primary_exc).__name__
                    cautions = list(analysis.get("cautions") or [])
                    cautions.append(
                        "Model-backed analysis was unavailable for this run; local analysis was used."
                    )
                    analysis["cautions"] = cautions
            else:
                analysis = local_analyze_video(
                    str(video.get("title") or video_id), str(transcript["text"]), profile
                )
            analysis.update(
                {
                    "video_id": video_id,
                    "title": video.get("title"),
                    "url": video.get("url"),
                    "transcript_sha256": transcript_hash,
                    "profile_sha256": profile_hash,
                    "analyzed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            workspace.save_analysis(video_id, analysis)
            counts["analyzed"] += 1
            status = str(analysis["mode"])
        except Exception as exc:
            counts["failed"] += 1
            status = "failed"
            workspace.append_error(
                {
                    "stage": "analysis",
                    "video_id": video_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )
        if on_progress:
            on_progress(index, len(videos), video_id, status)
    return counts


def _build_concepts(workspace: Workspace) -> list[dict[str, Any]]:
    concepts: list[dict[str, Any]] = []
    for video in workspace.load_manifest():
        video_id = str(video["id"])
        analysis = workspace.load_analysis(video_id)
        if not analysis:
            continue
        for index, concept in enumerate(analysis.get("concepts") or []):
            claim = str(concept.get("claim") or "").strip()
            if not claim:
                continue
            item = dict(concept)
            item.update(
                {
                    "concept_id": f"{video_id}:{index}",
                    "video_id": video_id,
                    "video_title": video.get("title"),
                    "claim": claim,
                }
            )
            concepts.append(item)
    return concepts


def _local_embeddings(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=float)
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.97, sublinear_tf=True
    )
    try:
        sparse = vectorizer.fit_transform(texts)
    except ValueError:
        # A corpus made entirely of stop words is unusual but should not crash a resumable run.
        sparse = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)

    n_samples, n_features = sparse.shape
    dimensions = min(256, n_samples - 1, n_features - 1)
    if dimensions < 2:
        dense = sparse.toarray().astype(float)
        return np.asarray(Normalizer(copy=False).fit_transform(dense), dtype=float)
    pipeline = make_pipeline(TruncatedSVD(dimensions, random_state=42), Normalizer(copy=False))
    return np.asarray(pipeline.fit_transform(sparse), dtype=float)


def _top_label(texts: list[str], maximum_words: int = 5) -> str:
    stop = {
        "the", "and", "that", "this", "with", "from", "your", "you", "for", "are", "into",
        "have", "will", "can", "when", "what", "how", "their", "they", "more", "than", "about",
        "need", "should", "something", "things", "because", "which", "being", "really", "very",
    }
    tokens: Counter[str] = Counter()
    for text in texts:
        words = re.findall(r"[a-z][a-z-]{2,}", text.lower())
        tokens.update(word for word in set(words) if word not in stop)
    label = " / ".join(word for word, _ in tokens.most_common(maximum_words))
    return label or "miscellaneous idea"


def cluster_and_select(
    workspace: Workspace,
    *,
    target_coverage: float = 0.8,
    max_minutes: float | None = 120.0,
    max_fraction: float = 0.2,
    similarity_threshold: float = 0.76,
    embedding_mode: str = "auto",
    embedding_model: str = "text-embedding-3-small",
    model: str | None = None,
) -> dict[str, Any]:
    concepts = _build_concepts(workspace)
    if not concepts:
        raise RuntimeError("No analyzed concepts found. Run `channel-compressor analyze` first.")
    embedding_mode = embedding_mode.lower()
    if embedding_mode not in {"auto", "openai", "local"}:
        raise ValueError("Embedding mode must be one of: auto, openai, local")
    target_coverage = clamp(target_coverage, 0.01, 1.0)
    max_fraction = clamp(max_fraction, 0.01, 1.0)
    texts = [str(item["claim"]) for item in concepts]
    use_openai = embedding_mode == "openai" or (
        embedding_mode == "auto" and bool(os.getenv("OPENAI_API_KEY"))
    )
    if use_openai:
        analyzer = OpenAIAnalyzer(model=model)
        vectors = np.asarray(analyzer.embed(texts, model=embedding_model), dtype=float)
        vector_source = embedding_model
    else:
        vectors = _local_embeddings(texts)
        vector_source = "tfidf-svd"
    if vectors.ndim != 2 or vectors.shape[0] != len(concepts):
        raise RuntimeError("Embedding provider returned an invalid matrix")
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    if vectors.shape[1] == 0:
        vectors = np.ones((len(concepts), 1), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero_rows = norms[:, 0] <= 1e-12
    if np.any(zero_rows):
        vectors[zero_rows, 0] = 1e-12
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    if len(concepts) == 1:
        labels = np.asarray([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - similarity_threshold,
        )
        labels = clustering.fit_predict(vectors)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)

    clusters: list[dict[str, Any]] = []
    concept_to_cluster: dict[str, str] = {}
    for ordinal, (_, indexes) in enumerate(sorted(grouped.items()), start=1):
        members = [concepts[index] for index in indexes]
        unique_videos = sorted({str(item["video_id"]) for item in members})
        utilities: list[float] = []
        kind_weights: list[float] = []
        for item in members:
            kind_weight = KIND_WEIGHT.get(str(item.get("kind")), 0.8)
            kind_weights.append(kind_weight)
            utility = (
                safe_float(item.get("salience"), 0.5)
                * (0.28 + 0.72 * safe_float(item.get("viewer_relevance"), 0.5))
                * (0.45 + 0.55 * safe_float(item.get("actionability"), 0.5))
                * (0.55 + 0.45 * safe_float(item.get("epistemic_quality"), 0.5))
                * (0.72 + 0.28 * safe_float(item.get("specificity"), 0.5))
                * kind_weight
            )
            utilities.append(utility)
        # Repetition is a weak centrality signal, not a license for a generic trope to dominate.
        prevalence_boost = min(1.35, 1.0 + 0.06 * math.log2(1.0 + len(unique_videos)))
        importance = max(utilities) * prevalence_boost
        cluster_id = f"c{ordinal:04d}"
        for item in members:
            concept_to_cluster[str(item["concept_id"])] = cluster_id
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": _top_label([str(item["claim"]) for item in members]),
                "importance": float(importance),
                "video_count": len(unique_videos),
                "member_count": len(members),
                "needs_verification_share": sum(bool(item.get("needs_verification")) for item in members)
                / len(members),
                "member_concepts": [str(item["concept_id"]) for item in members],
                "example_claims": [str(item["claim"]) for item in sorted(
                    members,
                    key=lambda item: safe_float(item.get("salience"), 0.5),
                    reverse=True,
                )[:3]],
            }
        )

    cluster_by_id = {str(item["cluster_id"]): item for item in clusters}
    manifest = workspace.manifest_by_id()
    video_contrib: dict[str, dict[str, float]] = defaultdict(dict)
    for item in concepts:
        concept_id = str(item["concept_id"])
        cluster_id = concept_to_cluster[concept_id]
        utility = (
            safe_float(item.get("salience"), 0.5)
            * (0.28 + 0.72 * safe_float(item.get("viewer_relevance"), 0.5))
            * (0.45 + 0.55 * safe_float(item.get("actionability"), 0.5))
            * (0.55 + 0.45 * safe_float(item.get("epistemic_quality"), 0.5))
            * KIND_WEIGHT.get(str(item.get("kind")), 0.8)
        )
        video_id = str(item["video_id"])
        video_contrib[video_id][cluster_id] = max(
            video_contrib[video_id].get(cluster_id, 0.0), float(utility)
        )

    # Normalize each cluster so one video can fully cover it. Cluster weight carries importance.
    cluster_max: dict[str, float] = {}
    for cluster_id in cluster_by_id:
        cluster_max[cluster_id] = max(
            (contrib.get(cluster_id, 0.0) for contrib in video_contrib.values()), default=1.0
        ) or 1.0
    normalized: dict[str, dict[str, float]] = {
        video_id: {
            cluster_id: value / cluster_max[cluster_id]
            for cluster_id, value in contributions.items()
        }
        for video_id, contributions in video_contrib.items()
    }
    weights = {cluster_id: float(item["importance"]) for cluster_id, item in cluster_by_id.items()}
    total_weight = sum(weights.values()) or 1.0

    def duration_minutes(video_id: str) -> float:
        video = manifest.get(video_id, {})
        duration = safe_float(video.get("duration_seconds"), 0.0)
        if duration <= 0:
            transcript = workspace.load_transcript(video_id) or {}
            segments = transcript.get("segments") or []
            if segments:
                last = segments[-1]
                duration = safe_float(last.get("start"), 0.0) + safe_float(last.get("duration"), 0.0)
            if duration <= 0:
                duration = safe_float(transcript.get("word_count"), 0.0) / 145.0 * 60.0
        return max(0.5, duration / 60.0)

    candidates = [video_id for video_id in normalized if video_id in manifest]
    max_videos = max(1, math.ceil(len(candidates) * max_fraction))
    selected: list[dict[str, Any]] = []
    covered = {cluster_id: 0.0 for cluster_id in weights}
    used_minutes = 0.0
    remaining = set(candidates)

    while remaining and len(selected) < max_videos:
        best: tuple[float, float, str, dict[str, float]] | None = None
        for video_id in remaining:
            minutes = duration_minutes(video_id)
            if max_minutes is not None and used_minutes + minutes > max_minutes:
                continue
            marginal_by_cluster = {
                cluster_id: max(0.0, value - covered[cluster_id])
                for cluster_id, value in normalized[video_id].items()
                if value > covered[cluster_id]
            }
            marginal = sum(weights[cid] * value for cid, value in marginal_by_cluster.items())
            analysis = workspace.load_analysis(video_id) or {}
            audience_fit = safe_float(analysis.get("audience_fit"), 0.5)
            epistemic_penalty = np.mean(
                [
                    1.0 - 0.35 * bool(concept.get("needs_verification"))
                    for concept in analysis.get("concepts") or []
                ]
                or [1.0]
            )
            score = (
                marginal
                * (0.72 + 0.28 * audience_fit)
                * float(epistemic_penalty)
                / (minutes ** 0.78)
            )
            if best is None or score > best[0]:
                best = (float(score), float(marginal), video_id, marginal_by_cluster)
        if best is None or best[1] <= 1e-9:
            break
        score, marginal, video_id, marginal_by_cluster = best
        minutes = duration_minutes(video_id)
        for cluster_id, value in normalized[video_id].items():
            covered[cluster_id] = max(covered[cluster_id], value)
        used_minutes += minutes
        cumulative = sum(weights[cid] * value for cid, value in covered.items()) / total_weight
        total_video_value = sum(weights[cid] * value for cid, value in normalized[video_id].items())
        unique_clusters = sorted(
            marginal_by_cluster,
            key=lambda cid: weights[cid] * marginal_by_cluster[cid],
            reverse=True,
        )
        analysis = workspace.load_analysis(video_id) or {}
        compressibility = safe_float(analysis.get("compressibility"), 0.7)
        watchability = safe_float(analysis.get("watchability"), 0.4)
        if watchability >= 0.68 and compressibility <= 0.62:
            consume_mode = "watch"
        elif watchability >= 0.50 and compressibility <= 0.78:
            consume_mode = "watch key sections"
        else:
            consume_mode = "read transcript/summary"
        selected.append(
            {
                "rank": len(selected) + 1,
                "video_id": video_id,
                "minutes": minutes,
                "marginal_value": marginal,
                "marginal_share": marginal / total_weight,
                "cumulative_coverage": cumulative,
                "score_per_minute": score,
                "novelty_share": marginal / max(total_video_value, 1e-9),
                "consume_mode": consume_mode,
                "unique_cluster_ids": unique_clusters[:5],
                "unique_cluster_labels": [cluster_by_id[cid]["label"] for cid in unique_clusters[:5]],
            }
        )
        remaining.remove(video_id)
        if cumulative >= target_coverage:
            break

    selected_ids = {str(item["video_id"]) for item in selected}
    # Rank leftovers by standalone value and identify the selected video with greatest overlap.
    leftovers: list[dict[str, Any]] = []
    for video_id in candidates:
        if video_id in selected_ids:
            continue
        standalone = sum(weights[cid] * value for cid, value in normalized[video_id].items())
        union_overlap_num = sum(
            weights[cid] * min(value, covered.get(cid, 0.0))
            for cid, value in normalized[video_id].items()
        )
        union_overlap_share = union_overlap_num / max(standalone, 1e-9)
        best_overlap = (0.0, None)
        for chosen in selected_ids:
            overlap_num = sum(
                weights[cid] * min(value, normalized.get(chosen, {}).get(cid, 0.0))
                for cid, value in normalized[video_id].items()
            )
            overlap_share = overlap_num / max(standalone, 1e-9)
            if overlap_share > best_overlap[0]:
                best_overlap = (overlap_share, chosen)
        leftovers.append(
            {
                "video_id": video_id,
                "standalone_value": standalone,
                "minutes": duration_minutes(video_id),
                "covered_by_selected_share": union_overlap_share,
                "most_redundant_with": best_overlap[1],
                "closest_selected_overlap_share": best_overlap[0],
            }
        )
    leftovers.sort(key=lambda item: item["standalone_value"] / item["minutes"], reverse=True)

    for concept in concepts:
        concept["cluster_id"] = concept_to_cluster[str(concept["concept_id"])]
    write_jsonl(workspace.concepts_path, concepts)
    atomic_write_json(
        workspace.clusters_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_source": vector_source,
            "similarity_threshold": similarity_threshold,
            "clusters": sorted(clusters, key=lambda item: item["importance"], reverse=True),
        },
    )
    achieved_coverage = selected[-1]["cumulative_coverage"] if selected else 0.0
    if achieved_coverage >= target_coverage:
        stop_reason = "target_coverage_reached"
    elif len(selected) >= max_videos:
        stop_reason = "max_fraction_reached"
    elif max_minutes is not None and all(
        used_minutes + duration_minutes(video_id) > max_minutes for video_id in remaining
    ):
        stop_reason = "time_budget_reached"
    else:
        stop_reason = "no_additional_value"

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_coverage": target_coverage,
        "achieved_coverage": achieved_coverage,
        "stop_reason": stop_reason,
        "max_minutes": max_minutes,
        "selected_minutes": used_minutes,
        "max_fraction": max_fraction,
        "max_videos": max_videos,
        "eligible_video_count": len(candidates),
        "selected": selected,
        "leftovers": leftovers,
    }
    atomic_write_json(workspace.selection_path, result)
    return result
