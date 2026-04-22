from __future__ import annotations

from pathlib import Path

from .config_loader import (
    load_capability_definitions,
    load_model_tiers,
    load_task_profiles,
)
from .intent.models import CapabilityResolution
from .intent.resolver import IntentResolver, build_request_context
from .models import (
    FilteredCandidate,
    ModelSelection,
    ModelTier,
    RankedCandidate,
    RequestContext,
    ScoreComponent,
    ScoringWeights,
    SelectionDecision,
    TaskProfile,
)


class SelectionError(RuntimeError):
    pass


class DeterministicSelector:
    def __init__(
        self,
        models: tuple[ModelTier, ...],
        profiles: dict[str, TaskProfile],
        intent_resolver: IntentResolver | None = None,
    ) -> None:
        self._models = models
        self._profiles = profiles
        self._intent_resolver = intent_resolver

    @classmethod
    def from_yaml(
        cls,
        models_path: str | Path,
        task_profiles_path: str | Path,
        capabilities_path: str | Path | None = None,
    ) -> "DeterministicSelector":
        models = load_model_tiers(models_path)
        profiles = load_task_profiles(task_profiles_path)
        intent_resolver = None
        if capabilities_path is not None:
            intent_resolver = IntentResolver(load_capability_definitions(capabilities_path))
        return cls(models=models, profiles=profiles, intent_resolver=intent_resolver)

    def select(self, context: RequestContext) -> SelectionDecision:
        profile = self._resolve_profile(context.capability)
        debug: list[str] = [f"profile={profile.capability}"]
        filtered_candidates: list[FilteredCandidate] = []

        ranked_pool: list[
            tuple[ModelTier, float, tuple[ScoreComponent, ...], tuple[str, ...]]
        ] = []

        for model in self._models:
            compatible, filter_reasons = self._is_compatible(model, context, profile)
            if not compatible:
                debug.append(
                    f"filtered:{model.selection_tier}:{'|'.join(filter_reasons)}"
                )
                filtered_candidates.append(
                    FilteredCandidate(
                        selection_tier=model.selection_tier,
                        reasons=filter_reasons,
                    )
                )
                continue

            score, components, reasons = self._score(model, context, profile)
            ranked_pool.append((model, score, components, reasons))

        if not ranked_pool:
            raise SelectionError(
                f"No compatible model candidates for capability '{context.capability}'"
            )

        ranked_pool.sort(key=lambda item: (-item[1], item[0].selection_tier))

        ranked_candidates = tuple(
            RankedCandidate(
                selection_tier=model.selection_tier,
                provider=model.provider,
                model_name=model.model_name,
                deployment_name=model.deployment_name,
                score=round(score, 6),
                score_components=components,
                reasons=reasons,
            )
            for model, score, components, reasons in ranked_pool
        )

        fallback_count = max(0, min(profile.fallback_count, len(ranked_candidates) - 1))
        primary = self._selection_from_candidate(ranked_candidates[0])
        fallbacks = tuple(
            self._selection_from_candidate(candidate)
            for candidate in ranked_candidates[1 : 1 + fallback_count]
        )

        return SelectionDecision(
            capability=context.capability,
            profile=profile.capability,
            primary=primary,
            fallbacks=fallbacks,
            ranked_candidates=ranked_candidates,
            filtered_candidates=tuple(filtered_candidates),
            debug_reasons=tuple(debug),
        )

    def select_prompt(self, prompt: str) -> SelectionDecision:
        resolution = self.resolve_intent(prompt)
        return self.select(build_request_context(resolution))

    def infer_context(self, prompt: str) -> RequestContext:
        return build_request_context(self.resolve_intent(prompt))

    def resolve_intent(self, prompt: str) -> CapabilityResolution:
        if self._intent_resolver is None:
            raise SelectionError(
                "Prompt selection requires an IntentResolver. "
                "Pass capabilities_path to from_yaml(...) or use IntentResolver directly."
            )
        return self._intent_resolver.resolve(prompt)

    def _selection_from_candidate(self, candidate: RankedCandidate) -> ModelSelection:
        return ModelSelection(
            selection_tier=candidate.selection_tier,
            provider=candidate.provider,
            model_name=candidate.model_name,
            deployment_name=candidate.deployment_name,
        )

    def _resolve_profile(self, capability: str) -> TaskProfile:
        if capability in self._profiles:
            return self._profiles[capability]
        if "default" in self._profiles:
            return self._profiles["default"]
        raise SelectionError(
            f"No profile found for capability '{capability}' and no 'default' profile present"
        )

    def _is_compatible(
        self,
        model: ModelTier,
        context: RequestContext,
        profile: TaskProfile,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []

        if not model.enabled:
            reasons.append("disabled")

        if context.needs_tools and not model.supports_tools:
            reasons.append("needs_tools")
        if context.needs_json and not model.supports_json:
            reasons.append("needs_json")
        if context.long_context and not model.long_context:
            reasons.append("needs_long_context")

        req = profile.required_constraints
        if req.supports_tools is True and not model.supports_tools:
            reasons.append("profile_requires_tools")
        if req.supports_json is True and not model.supports_json:
            reasons.append("profile_requires_json")
        if req.long_context is True and not model.long_context:
            reasons.append("profile_requires_long_context")

        missing_tags = [tag for tag in req.required_role_tags if tag not in model.role_tags]
        if missing_tags:
            reasons.append(f"missing_role_tags:{','.join(sorted(missing_tags))}")

        return (len(reasons) == 0, tuple(reasons))

    def _adjust_weights(
        self,
        profile: TaskProfile,
        context: RequestContext,
    ) -> tuple[ScoringWeights, list[ScoreComponent], list[str]]:
        adjusted = profile.scoring_weights
        components: list[ScoreComponent] = []
        reasons: list[str] = []

        priority_adjustment = profile.priority_weight_adjustments.get(context.priority)
        if priority_adjustment is not None:
            adjusted = self._add_weights(adjusted, priority_adjustment)
            components.append(
                ScoreComponent(
                    name=f"priority:{context.priority}",
                    value=0.0,
                    details=self._format_adjustment(priority_adjustment),
                )
            )
            reasons.append(
                f"priority_{context.priority}:{self._format_adjustment(priority_adjustment)}"
            )

        budget_adjustment = profile.budget_weight_adjustments.get(context.budget_mode)
        if budget_adjustment is not None:
            adjusted = self._add_weights(adjusted, budget_adjustment)
            components.append(
                ScoreComponent(
                    name=f"budget:{context.budget_mode}",
                    value=0.0,
                    details=self._format_adjustment(budget_adjustment),
                )
            )
            reasons.append(
                f"budget_{context.budget_mode}:{self._format_adjustment(budget_adjustment)}"
            )

        adjusted = ScoringWeights(
            coding=max(0.0, adjusted.coding),
            reasoning=max(0.0, adjusted.reasoning),
            latency=max(0.0, adjusted.latency),
            cost=max(0.0, adjusted.cost),
            reliability=max(0.0, adjusted.reliability),
        )

        return adjusted, components, reasons

    def _score(
        self,
        model: ModelTier,
        context: RequestContext,
        profile: TaskProfile,
    ) -> tuple[float, tuple[ScoreComponent, ...], tuple[str, ...]]:
        components: list[ScoreComponent] = []
        reasons: list[str] = []
        weights, adjustment_components, adjusted_reasons = self._adjust_weights(
            profile,
            context,
        )
        components.extend(adjustment_components)
        reasons.extend(adjusted_reasons)

        score = 0.0

        score += self._weighted_component(
            components, reasons, "coding", weights.coding, model.coding_score
        )
        score += self._weighted_component(
            components, reasons, "reasoning", weights.reasoning, model.reasoning_score
        )
        score += self._weighted_component(
            components, reasons, "latency", weights.latency, model.latency_score
        )
        score += self._weighted_component(
            components, reasons, "cost", weights.cost, model.cost_score
        )
        score += self._weighted_component(
            components,
            reasons,
            "reliability",
            weights.reliability,
            model.reliability_score,
        )

        if model.selection_tier in profile.preferred_tiers:
            score += profile.boosts.preferred_tier
            components.append(
                ScoreComponent(
                    name="preferred_tier",
                    value=profile.boosts.preferred_tier,
                    details=model.selection_tier,
                )
            )
            reasons.append(f"preferred_tier:+{profile.boosts.preferred_tier:.2f}")

        if any(tag in profile.preferred_tags for tag in model.role_tags):
            score += profile.boosts.preferred_tag
            matched_tags = sorted(set(profile.preferred_tags).intersection(model.role_tags))
            components.append(
                ScoreComponent(
                    name="preferred_tag",
                    value=profile.boosts.preferred_tag,
                    details=",".join(matched_tags),
                )
            )
            reasons.append(f"preferred_tag:+{profile.boosts.preferred_tag:.2f}")

        if (
            context.retry_count > 0
            and profile.escalation.premium_on_retry
            and (
                "premium" in model.role_tags
                or model.reasoning_score >= profile.escalation.retry_reasoning_threshold
            )
        ):
            retry_boost = profile.boosts.retry_premium * float(context.retry_count)
            score += retry_boost
            components.append(
                ScoreComponent(
                    name="retry_escalation",
                    value=retry_boost,
                    details=f"retry_count={context.retry_count}",
                )
            )
            reasons.append(f"retry_escalation:+{retry_boost:.2f}")

        if (
            profile.penalties.low_reliability_threshold > 0
            and model.reliability_score < profile.penalties.low_reliability_threshold
        ):
            score -= profile.penalties.low_reliability
            components.append(
                ScoreComponent(
                    name="reliability_penalty",
                    value=-profile.penalties.low_reliability,
                    details=(
                        f"{model.reliability_score:.2f}<"
                        f"{profile.penalties.low_reliability_threshold:.2f}"
                    ),
                )
            )
            reasons.append(f"reliability_penalty:-{profile.penalties.low_reliability:.2f}")

        return score, tuple(components), tuple(reasons)

    def _weighted_component(
        self,
        components: list[ScoreComponent],
        reasons: list[str],
        name: str,
        weight: float,
        model_score: float,
    ) -> float:
        value = weight * model_score
        components.append(
            ScoreComponent(
                name=name,
                value=value,
                details=f"{weight:.2f}*{model_score:.2f}",
            )
        )
        reasons.append(f"{name}:{weight:.2f}*{model_score:.2f}={value:.2f}")
        return value

    def _add_weights(
        self,
        left: ScoringWeights,
        right: ScoringWeights,
    ) -> ScoringWeights:
        return ScoringWeights(
            coding=left.coding + right.coding,
            reasoning=left.reasoning + right.reasoning,
            latency=left.latency + right.latency,
            cost=left.cost + right.cost,
            reliability=left.reliability + right.reliability,
        )

    def _format_adjustment(self, weights: ScoringWeights) -> str:
        parts = []
        for name, value in (
            ("coding", weights.coding),
            ("reasoning", weights.reasoning),
            ("latency", weights.latency),
            ("cost", weights.cost),
            ("reliability", weights.reliability),
        ):
            if value != 0:
                parts.append(f"{name}:{value:+.2f}")
        return ",".join(parts) if parts else "no-op"
