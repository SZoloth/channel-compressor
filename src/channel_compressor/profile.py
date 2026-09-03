from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_profile(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Profile YAML must contain an object at the top level")
    return value


def profile_as_prompt(profile: dict[str, Any]) -> str:
    if not profile:
        return "No personal profile was supplied; judge broad durable usefulness."
    lines: list[str] = []
    name = profile.get("name")
    if name:
        lines.append(f"Viewer: {name}")
    for key, heading in (
        ("priorities", "Priorities"),
        ("deprioritize", "Deprioritize"),
        ("preferences", "Preferences"),
    ):
        value = profile.get(key)
        if not value:
            continue
        lines.append(f"{heading}:")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    label = item.get("topic") or item.get("name") or str(item)
                    weight = item.get("weight")
                    lines.append(f"- {label}" + (f" (weight {weight})" if weight is not None else ""))
                else:
                    lines.append(f"- {item}")
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                lines.append(f"- {subkey}: {subvalue}")
        else:
            lines.append(f"- {value}")
    return "\n".join(lines)


def profile_keywords(profile: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    positive: dict[str, float] = {}
    negative: dict[str, float] = {}
    for item in profile.get("priorities") or []:
        if isinstance(item, dict):
            weight = float(item.get("weight", 1.0))
            terms = item.get("keywords") or [item.get("topic", "")]
        else:
            weight = 1.0
            terms = [str(item)]
        for term in terms:
            for token in str(term).lower().replace("/", " ").replace("-", " ").split():
                if len(token) >= 3:
                    positive[token] = max(positive.get(token, 0.0), weight)
    for item in profile.get("deprioritize") or []:
        if isinstance(item, dict):
            weight = float(item.get("weight", 1.0))
            terms = item.get("keywords") or [item.get("topic", "")]
        else:
            weight = 1.0
            terms = [str(item)]
        for term in terms:
            for token in str(term).lower().replace("/", " ").replace("-", " ").split():
                if len(token) >= 3:
                    negative[token] = max(negative.get(token, 0.0), weight)
    return positive, negative
