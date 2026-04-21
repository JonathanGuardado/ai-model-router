from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config_loader import load_model_tiers, load_task_profiles
from .models import (
    ModelTier,
    RankedCandidate,
    RequestContext,
    RoutingDecision,
    ScoringWeights,
    TaskProfile,
)


class RoutingError(RuntimeError):
    pass


class DeterministicRouter:
    def __init__(self, models: tuple[ModelTier, ...], profiles: dict[str, TaskProfile]) -> None:
        self._models = models
        self._profiles = profiles

    @classmethod
    def from_yaml(
        cls,
        models_path: str | Path,
        task_profiles_path: str | Path,
    ) -> "DeterministicRouter":
        models = load_model_tiers(models_path)
        profiles = load_task_profiles(task_profiles_path)
        return cls(models=models, profiles=profiles)

    def route(self, context: RequestContext) -> RoutingDecision:
        profile = self._resolve_profile(context.capability)
        debug: list[str] = [f"profile={profile.capability}"]

        ranked_pool: list[tuple[ModelTier, float, tuple[str, ...]]] = []

        for model in self._models:
            compatible, filter_reasons = self._is_compatible(model, context, profile)
            if not compatible:
                debug.append(f"filtered:{model.name}:{'|'.join(filter_reasons)}")
                continue

            score, reasons = self._score(model, context, profile)
            ranked_pool.append((model, score, reasons))

        if not ranked_pool:
            raise RoutingError(
                f"No compatible model candidates for capability '{context.capability}'"
            )

        ranked_pool.sort(key=lambda item: (-item[1], item[0].name))

        ranked_candidates = tuple(
            RankedCandidate(
                model_name=model.name,
                provider=model.provider,
                deployment_name=model.deployment_name,
                score=round(score, 6),
                reasons=reasons,
            )
            for model, score, reasons in ranked_pool
        )

        fallback_count = max(0, min(profile.fallback_count, len(ranked_candidates) - 1))
        fallback_models = tuple(
            candidate.model_name for candidate in ranked_candidates[1 : 1 + fallback_count]
        )

        return RoutingDecision(
            primary_model=ranked_candidates[0].model_name,
            fallback_models=fallback_models,
            ranked_candidates=ranked_candidates,
            debug_reasons=tuple(debug),
        )

    def _resolve_profile(self, capability: str) -> TaskProfile:
        if capability in self._profiles:
            return self._profiles[capability]
        if "default" in self._profiles:
            return self._profiles["default"]
        raise RoutingError(
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
        base: ScoringWeights,
        context: RequestContext,
    ) -> tuple[ScoringWeights, list[str]]:
        adjusted = replace(base)
        reasons: list[str] = []

        if context.priority == "latency":
            adjusted = replace(adjusted, latency=adjusted.latency + 1.5)
            reasons.append("priority_latency:+latency")
        elif context.priority == "quality":
            adjusted = replace(
                adjusted,
                coding=adjusted.coding + 0.8,
                reasoning=adjusted.reasoning + 1.0,
                reliability=adjusted.reliability + 0.6,
            )
            reasons.append("priority_quality:+coding,+reasoning,+reliability")
        elif context.priority == "reliability":
            adjusted = replace(adjusted, reliability=adjusted.reliability + 1.5)
            reasons.append("priority_reliability:+reliability")

        if context.budget_mode == "economy":
            adjusted = replace(
                adjusted,
                cost=adjusted.cost + 1.8,
                latency=adjusted.latency + 0.4,
                reasoning=max(0.0, adjusted.reasoning - 0.3),
            )
            reasons.append("budget_economy:+cost,+latency,-reasoning")
        elif context.budget_mode == "premium":
            adjusted = replace(
                adjusted,
                reasoning=adjusted.reasoning + 1.1,
                reliability=adjusted.reliability + 0.8,
                cost=max(0.0, adjusted.cost - 0.5),
            )
            reasons.append("budget_premium:+reasoning,+reliability,-cost")

        return adjusted, reasons

    def _score(
        self,
        model: ModelTier,
        context: RequestContext,
        profile: TaskProfile,
    ) -> tuple[float, tuple[str, ...]]:
        reasons: list[str] = []
        weights, adjusted_reasons = self._adjust_weights(profile.scoring_weights, context)
        reasons.extend(adjusted_reasons)

        score = 0.0

        score += weights.coding * model.coding_score
        reasons.append(
            f"coding:{weights.coding:.2f}*{model.coding_score:.2f}={weights.coding * model.coding_score:.2f}"
        )

        score += weights.reasoning * model.reasoning_score
        reasons.append(
            f"reasoning:{weights.reasoning:.2f}*{model.reasoning_score:.2f}={weights.reasoning * model.reasoning_score:.2f}"
        )

        score += weights.latency * model.latency_score
        reasons.append(
            f"latency:{weights.latency:.2f}*{model.latency_score:.2f}={weights.latency * model.latency_score:.2f}"
        )

        score += weights.cost * model.cost_score
        reasons.append(
            f"cost:{weights.cost:.2f}*{model.cost_score:.2f}={weights.cost * model.cost_score:.2f}"
        )

        score += weights.reliability * model.reliability_score
        reasons.append(
            f"reliability:{weights.reliability:.2f}*{model.reliability_score:.2f}={weights.reliability * model.reliability_score:.2f}"
        )

        if model.name in profile.preferred_tiers:
            score += profile.boosts.preferred_tier
            reasons.append(f"preferred_tier:+{profile.boosts.preferred_tier:.2f}")

        if any(tag in profile.preferred_tags for tag in model.role_tags):
            score += profile.boosts.preferred_tag
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
            reasons.append(f"retry_escalation:+{retry_boost:.2f}")

        if (
            profile.penalties.low_reliability_threshold > 0
            and model.reliability_score < profile.penalties.low_reliability_threshold
        ):
            score -= profile.penalties.low_reliability
            reasons.append(f"reliability_penalty:-{profile.penalties.low_reliability:.2f}")

        return score, tuple(reasons)
