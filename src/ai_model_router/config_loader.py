from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    Boosts,
    Escalation,
    ModelTier,
    Penalties,
    RequiredConstraints,
    ScoringWeights,
    TaskProfile,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"YAML root must be a mapping: {path}"
        raise ValueError(msg)
    return data


def load_model_tiers(path: str | Path) -> tuple[ModelTier, ...]:
    data = _read_yaml(path)
    raw_models = data.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("'models' must be a list")

    models: list[ModelTier] = []
    seen_names: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be a mapping")

        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate model name: {name}")
        seen_names.add(name)

        models.append(
            ModelTier(
                name=name,
                provider=str(item["provider"]),
                deployment_name=str(item["deployment_name"]),
                role_tags=tuple(item.get("role_tags", [])),
                supports_tools=bool(item.get("supports_tools", False)),
                supports_json=bool(item.get("supports_json", False)),
                long_context=bool(item.get("long_context", False)),
                coding_score=float(item.get("coding_score", 0)),
                reasoning_score=float(item.get("reasoning_score", 0)),
                latency_score=float(item.get("latency_score", 0)),
                cost_score=float(item.get("cost_score", 0)),
                reliability_score=float(item.get("reliability_score", 0)),
                enabled=bool(item.get("enabled", True)),
            )
        )

    return tuple(models)


def load_task_profiles(path: str | Path) -> dict[str, TaskProfile]:
    data = _read_yaml(path)
    raw_profiles = data.get("task_profiles", [])
    if not isinstance(raw_profiles, list):
        raise ValueError("'task_profiles' must be a list")

    profiles: dict[str, TaskProfile] = {}
    for item in raw_profiles:
        if not isinstance(item, dict):
            raise ValueError("Each task profile entry must be a mapping")

        capability = str(item["capability"])
        if capability in profiles:
            raise ValueError(f"Duplicate capability profile: {capability}")

        required = item.get("required_constraints", {}) or {}
        scoring = item.get("scoring_weights", {}) or {}
        boosts = item.get("boosts", {}) or {}
        penalties = item.get("penalties", {}) or {}
        escalation = item.get("escalation", {}) or {}

        profile = TaskProfile(
            capability=capability,
            preferred_tiers=tuple(item.get("preferred_tiers", [])),
            preferred_tags=tuple(item.get("preferred_tags", [])),
            required_constraints=RequiredConstraints(
                supports_tools=required.get("supports_tools"),
                supports_json=required.get("supports_json"),
                long_context=required.get("long_context"),
                required_role_tags=tuple(required.get("required_role_tags", [])),
            ),
            scoring_weights=ScoringWeights(
                coding=float(scoring.get("coding", 0.0)),
                reasoning=float(scoring.get("reasoning", 0.0)),
                latency=float(scoring.get("latency", 0.0)),
                cost=float(scoring.get("cost", 0.0)),
                reliability=float(scoring.get("reliability", 0.0)),
            ),
            boosts=Boosts(
                preferred_tier=float(boosts.get("preferred_tier", 1.0)),
                preferred_tag=float(boosts.get("preferred_tag", 0.75)),
                retry_premium=float(boosts.get("retry_premium", 1.0)),
            ),
            penalties=Penalties(
                low_reliability_threshold=float(
                    penalties.get("low_reliability_threshold", 0.0)
                ),
                low_reliability=float(penalties.get("low_reliability", 0.0)),
            ),
            escalation=Escalation(
                premium_on_retry=bool(escalation.get("premium_on_retry", True)),
                retry_reasoning_threshold=float(
                    escalation.get("retry_reasoning_threshold", 8.0)
                ),
            ),
            fallback_count=int(item.get("fallback_count", 2)),
        )

        profiles[capability] = profile

    return profiles
