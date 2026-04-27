from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .intent.models import CapabilityDefinition
from .models import (
    BudgetMode,
    Boosts,
    Escalation,
    ModelTier,
    Penalties,
    Priority,
    RequiredConstraints,
    ScoringWeights,
    TaskProfile,
)

PRIORITIES: tuple[Priority, ...] = ("balanced", "latency", "quality", "reliability")
BUDGET_MODES: tuple[BudgetMode, ...] = ("balanced", "economy", "premium")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        msg = f"YAML root must be a mapping: {path}"
        raise ValueError(msg)
    return data


def _load_weights(raw: Any, *, context: str) -> ScoringWeights:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")

    return ScoringWeights(
        coding=float(raw.get("coding", 0.0)),
        reasoning=float(raw.get("reasoning", 0.0)),
        latency=float(raw.get("latency", 0.0)),
        cost=float(raw.get("cost", 0.0)),
        reliability=float(raw.get("reliability", 0.0)),
    )


def _load_weight_adjustments(
    raw: Any,
    *,
    allowed_keys: tuple[str, ...],
    context: str,
) -> dict[str, ScoringWeights]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")

    adjustments: dict[str, ScoringWeights] = {}
    for key, weights in raw.items():
        key_str = str(key)
        if key_str not in allowed_keys:
            raise ValueError(f"Unknown {context} key: {key_str}")
        adjustments[key_str] = _load_weights(weights, context=f"{context}.{key_str}")

    return adjustments


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

        selection_tier = str(item.get("selection_tier", item.get("name")))
        if selection_tier in seen_names:
            raise ValueError(f"Duplicate selection tier: {selection_tier}")
        seen_names.add(selection_tier)

        models.append(
            ModelTier(
                selection_tier=selection_tier,
                provider=str(item["provider"]),
                model_name=str(item.get("model_name", item.get("deployment_name"))),
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
                invocation=str(item.get("invocation", "openai_chat")),
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
        priority_adjustments = item.get("priority_weight_adjustments", {}) or {}
        budget_adjustments = item.get("budget_weight_adjustments", {}) or {}
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
            scoring_weights=_load_weights(
                scoring,
                context=f"task_profiles.{capability}.scoring_weights",
            ),
            priority_weight_adjustments=_load_weight_adjustments(
                priority_adjustments,
                allowed_keys=PRIORITIES,
                context=f"task_profiles.{capability}.priority_weight_adjustments",
            ),
            budget_weight_adjustments=_load_weight_adjustments(
                budget_adjustments,
                allowed_keys=BUDGET_MODES,
                context=f"task_profiles.{capability}.budget_weight_adjustments",
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


def load_capability_definitions(path: str | Path) -> tuple[CapabilityDefinition, ...]:
    data = _read_yaml(path)
    raw_capabilities = data.get("capabilities", [])
    if not isinstance(raw_capabilities, list):
        raise ValueError("'capabilities' must be a list")

    capabilities: list[CapabilityDefinition] = []
    seen_names: set[str] = set()
    for item in raw_capabilities:
        if not isinstance(item, dict):
            raise ValueError("Each capability entry must be a mapping")

        name = str(item["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate capability definition: {name}")
        seen_names.add(name)

        default_signals = item.get("default_signals", {}) or {}
        if not isinstance(default_signals, dict):
            raise ValueError(f"default_signals must be a mapping for capability: {name}")

        capabilities.append(
            CapabilityDefinition(
                name=name,
                description=str(item.get("description", "")),
                examples=tuple(str(value) for value in item.get("examples", [])),
                anti_examples=tuple(
                    str(value) for value in item.get("anti_examples", [])
                ),
                default_signals=dict(default_signals),
            )
        )

    return tuple(capabilities)
