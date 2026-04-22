from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_model_selector.models import BudgetMode, Priority, RequestContext

from .heuristics import StructuralHeuristics
from .models import CapabilityCandidate, CapabilityDefinition, CapabilityResolution
from .semantic_matcher import LocalSemanticMatcher, SemanticMatcher


class IntentResolver:
    def __init__(
        self,
        capabilities: tuple[CapabilityDefinition, ...],
        matcher: SemanticMatcher | None = None,
        heuristics: StructuralHeuristics | None = None,
        weak_threshold: float = 0.12,
        ambiguity_margin: float = 0.12,
        default_capability: str = "trivial.respond",
    ) -> None:
        self._capabilities = capabilities
        self._matcher = matcher or LocalSemanticMatcher()
        self._heuristics = heuristics or StructuralHeuristics()
        self._weak_threshold = weak_threshold
        self._ambiguity_margin = ambiguity_margin
        self._default_capability = default_capability
        self._capability_names = {capability.name for capability in capabilities}

    @classmethod
    def from_yaml(cls, capabilities_path: str | Path) -> "IntentResolver":
        from ai_model_selector.config_loader import load_capability_definitions

        return cls(load_capability_definitions(capabilities_path))

    def resolve(self, text: str) -> CapabilityResolution:
        semantic_candidates = self._matcher.match(text, self._capabilities)
        heuristic_result = self._heuristics.evaluate(text)
        candidates = self._apply_heuristics(
            semantic_candidates,
            heuristic_result.score_adjustments,
        )

        if not candidates:
            return self._default_resolution(
                signals=heuristic_result.signals,
                debug=("no_capabilities_loaded",),
            )

        top = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        weak = top.score < self._weak_threshold
        low_information = self._is_low_information(heuristic_result.signals)
        close = runner_up is not None and top.score - runner_up.score < self._ambiguity_margin
        ambiguous = weak or close or low_information
        capability = self._default_capability if weak else top.capability
        confidence = self._confidence(top, runner_up)

        if weak:
            source = "default"
        elif ambiguous:
            source = "ambiguous"
        else:
            source = "semantic"

        definition = self._definition_for(capability)
        signals = dict(definition.default_signals if definition else {})
        signals.update(heuristic_result.signals)

        debug = [
            f"top={top.capability}:{top.score:.3f}",
            f"confidence={confidence:.3f}",
        ]
        if runner_up:
            debug.append(f"runner_up={runner_up.capability}:{runner_up.score:.3f}")
        if weak:
            debug.append(f"weak_top_score:fallback={self._default_capability}")
        if close:
            debug.append("ambiguous_margin")
        if low_information:
            debug.append("low_information_input")
        debug.extend(heuristic_result.reasons)

        return CapabilityResolution(
            capability=capability,
            confidence=confidence,
            source=source,
            ambiguous=ambiguous,
            signals=signals,
            candidates=candidates,
            debug=tuple(debug),
        )

    def _apply_heuristics(
        self,
        candidates: tuple[CapabilityCandidate, ...],
        adjustments: dict[str, float],
    ) -> tuple[CapabilityCandidate, ...]:
        updated = []
        for candidate in candidates:
            adjustment = adjustments.get(candidate.capability, 0.0)
            reasons = list(candidate.reasons)
            if adjustment:
                reasons.append(f"structural_adjustment:+{adjustment:.3f}")
            updated.append(
                CapabilityCandidate(
                    capability=candidate.capability,
                    score=round(candidate.score + adjustment, 6),
                    reasons=tuple(reasons),
                )
            )
        return tuple(sorted(updated, key=lambda item: (-item.score, item.capability)))

    def _definition_for(self, capability: str) -> CapabilityDefinition | None:
        for definition in self._capabilities:
            if definition.name == capability:
                return definition
        return None

    def _confidence(
        self,
        top: CapabilityCandidate,
        runner_up: CapabilityCandidate | None,
    ) -> float:
        if runner_up is None:
            return min(1.0, top.score)
        return max(0.0, min(1.0, top.score - runner_up.score))

    def _is_low_information(self, signals: dict[str, Any]) -> bool:
        return (
            signals.get("very_short_input") is True
            and signals.get("looks_like_greeting") is not True
            and signals.get("implementation_style") is not True
            and signals.get("architecture_language") is not True
            and signals.get("contains_url") is not True
            and signals.get("contains_code_block") is not True
            and signals.get("contains_stack_trace") is not True
        )

    def _default_resolution(
        self,
        signals: dict[str, Any],
        debug: tuple[str, ...],
    ) -> CapabilityResolution:
        return CapabilityResolution(
            capability=self._default_capability,
            confidence=0.0,
            source="default",
            ambiguous=True,
            signals=signals,
            candidates=(),
            debug=debug,
        )


def build_request_context(
    resolution: CapabilityResolution,
    *,
    retry_count: int = 0,
) -> RequestContext:
    return RequestContext(
        capability=resolution.capability,
        retry_count=retry_count,
        needs_tools=bool(resolution.signals.get("needs_tools", False)),
        needs_json=bool(resolution.signals.get("needs_json", False)),
        long_context=bool(resolution.signals.get("long_context", False)),
        priority=_priority(resolution.signals.get("priority", "balanced")),
        budget_mode=_budget_mode(resolution.signals.get("budget_mode", "balanced")),
    )


def _priority(value: object) -> Priority:
    if value in {"balanced", "latency", "quality", "reliability"}:
        return value  # type: ignore[return-value]
    return "balanced"


def _budget_mode(value: object) -> BudgetMode:
    if value in {"balanced", "economy", "premium"}:
        return value  # type: ignore[return-value]
    return "balanced"
