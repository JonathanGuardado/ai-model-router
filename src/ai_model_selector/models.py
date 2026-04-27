from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Priority = Literal["balanced", "latency", "quality", "reliability"]
BudgetMode = Literal["balanced", "economy", "premium"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    capability: str
    retry_count: int = 0
    needs_tools: bool = False
    needs_json: bool = False
    long_context: bool = False
    priority: Priority = "balanced"
    budget_mode: BudgetMode = "balanced"


@dataclass(frozen=True, slots=True)
class ModelTier:
    selection_tier: str
    provider: str
    model_name: str
    deployment_name: str
    role_tags: tuple[str, ...]
    supports_tools: bool
    supports_json: bool
    long_context: bool
    coding_score: float
    reasoning_score: float
    latency_score: float
    cost_score: float
    reliability_score: float
    enabled: bool = True
    invocation: str = "openai_chat"


@dataclass(frozen=True, slots=True)
class RequiredConstraints:
    supports_tools: bool | None = None
    supports_json: bool | None = None
    long_context: bool | None = None
    required_role_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    coding: float = 0.0
    reasoning: float = 0.0
    latency: float = 0.0
    cost: float = 0.0
    reliability: float = 0.0


@dataclass(frozen=True, slots=True)
class Boosts:
    preferred_tier: float = 1.0
    preferred_tag: float = 0.75
    retry_premium: float = 1.0


@dataclass(frozen=True, slots=True)
class Penalties:
    low_reliability_threshold: float = 0.0
    low_reliability: float = 0.0


@dataclass(frozen=True, slots=True)
class Escalation:
    premium_on_retry: bool = True
    retry_reasoning_threshold: float = 8.0


@dataclass(frozen=True, slots=True)
class TaskProfile:
    capability: str
    preferred_tiers: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    required_constraints: RequiredConstraints = field(default_factory=RequiredConstraints)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    priority_weight_adjustments: dict[Priority, ScoringWeights] = field(default_factory=dict)
    budget_weight_adjustments: dict[BudgetMode, ScoringWeights] = field(default_factory=dict)
    boosts: Boosts = field(default_factory=Boosts)
    penalties: Penalties = field(default_factory=Penalties)
    escalation: Escalation = field(default_factory=Escalation)
    fallback_count: int = 2


@dataclass(frozen=True, slots=True)
class ModelSelection:
    selection_tier: str
    provider: str
    model_name: str
    deployment_name: str
    invocation: str = "openai_chat"


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    name: str
    value: float
    details: str


@dataclass(frozen=True, slots=True)
class FilteredCandidate:
    selection_tier: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    selection_tier: str
    provider: str
    model_name: str
    deployment_name: str
    score: float
    score_components: tuple[ScoreComponent, ...]
    reasons: tuple[str, ...]
    invocation: str = "openai_chat"


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    capability: str
    profile: str
    primary: ModelSelection
    fallbacks: tuple[ModelSelection, ...]
    ranked_candidates: tuple[RankedCandidate, ...]
    filtered_candidates: tuple[FilteredCandidate, ...]
    debug_reasons: tuple[str, ...]

    @property
    def primary_model(self) -> str:
        return self.primary.model_name

    @property
    def primary_selection_tier(self) -> str:
        return self.primary.selection_tier

    @property
    def fallback_models(self) -> tuple[str, ...]:
        return tuple(fallback.model_name for fallback in self.fallbacks)

    @property
    def fallback_selection_tiers(self) -> tuple[str, ...]:
        return tuple(fallback.selection_tier for fallback in self.fallbacks)
