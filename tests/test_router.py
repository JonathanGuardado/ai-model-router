from pathlib import Path

import pytest

from ai_model_router.config_loader import load_task_profiles
from ai_model_router.models import ModelTier, RequestContext, RequiredConstraints, TaskProfile
from ai_model_router.router import DeterministicRouter, RoutingError


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "config" / "models.yaml"
PROFILES = ROOT / "config" / "task_profiles.yaml"
CAPABILITIES = ROOT / "config" / "capabilities.yaml"


def _router() -> DeterministicRouter:
    return DeterministicRouter.from_yaml(MODELS, PROFILES)


def _router_with_prompt_helper() -> DeterministicRouter:
    return DeterministicRouter.from_yaml(MODELS, PROFILES, CAPABILITIES)


def test_route_with_explicit_trivial_capability_prefers_local_fast() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="trivial.classify",
            needs_json=True,
            budget_mode="economy",
        )
    )
    assert decision.primary_routing_tier == "local_fast"
    assert decision.primary_model == "local_model"
    assert decision.primary.provider == "local"
    assert len(decision.fallback_models) == 2


def test_route_prompt_convenience_uses_intent_resolver_for_hello() -> None:
    router = _router_with_prompt_helper()

    decision = router.route_prompt("hello")

    assert decision.capability == "trivial.respond"
    assert decision.primary_routing_tier == "local_fast"
    assert decision.primary_model == "local_model"


def test_route_prompt_convenience_uses_intent_resolver_for_code_request() -> None:
    router = _router_with_prompt_helper()

    decision = router.route_prompt("Write a Python function and tests for sorting users")

    assert decision.capability == "code.implement"
    assert decision.primary_routing_tier == "coding_primary"


def test_route_with_explicit_web_research_capability_requires_tools() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="web.research",
            needs_tools=True,
            long_context=True,
        )
    )
    assert decision.primary_routing_tier == "web_agent"
    assert "local_fast" not in [c.routing_tier for c in decision.ranked_candidates]
    assert any(c.routing_tier == "local_fast" for c in decision.filtered_candidates)


def test_route_with_explicit_code_capability_prefers_coding_primary() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="code.implement",
            needs_tools=True,
            needs_json=True,
            priority="quality",
        )
    )
    assert decision.primary_routing_tier == "coding_primary"
    assert decision.primary.deployment_name == "coding_model"


def test_architecture_design_retry_escalates_to_reasoning_primary() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="architecture.design",
            retry_count=1,
            long_context=True,
            priority="quality",
            budget_mode="premium",
        )
    )
    assert decision.primary_routing_tier == "reasoning_primary"


def test_deterministic_same_input_same_output() -> None:
    router = _router()
    context = RequestContext(
        capability="tests.analyze",
        needs_json=True,
        retry_count=1,
    )
    first = router.route(context)
    second = router.route(context)

    assert first.primary_routing_tier == second.primary_routing_tier
    assert first.fallback_routing_tiers == second.fallback_routing_tiers
    assert [c.routing_tier for c in first.ranked_candidates] == [
        c.routing_tier for c in second.ranked_candidates
    ]


def test_budget_economy_in_default_profile_prefers_local_fast() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="unknown.capability",
            budget_mode="economy",
            priority="latency",
        )
    )
    assert decision.primary_routing_tier == "local_fast"
    assert decision.profile == "default"


def test_score_breakdown_includes_configured_policy_adjustments() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="code.implement",
            needs_tools=True,
            needs_json=True,
            priority="quality",
            budget_mode="premium",
        )
    )

    primary = decision.ranked_candidates[0]
    component_names = [component.name for component in primary.score_components]

    assert "priority:quality" in component_names
    assert "budget:premium" in component_names
    assert "coding" in component_names
    assert any(reason.startswith("priority_quality:") for reason in primary.reasons)


def test_route_raises_when_no_compatible_model() -> None:
    router = DeterministicRouter(
        models=(
            ModelTier(
                routing_tier="plain_text_only",
                provider="local",
                model_name="plain-text-only",
                deployment_name="plain-text-only-v1",
                role_tags=(),
                supports_tools=False,
                supports_json=False,
                long_context=False,
                coding_score=1.0,
                reasoning_score=1.0,
                latency_score=1.0,
                cost_score=1.0,
                reliability_score=1.0,
            ),
        ),
        profiles={
            "default": TaskProfile(
                capability="default",
                required_constraints=RequiredConstraints(supports_json=True),
            )
        },
    )

    with pytest.raises(RoutingError):
        router.route(
            RequestContext(
                capability="web.research",
                needs_tools=True,
                long_context=True,
                needs_json=True,
            )
        )


def test_deterministic_tie_breaks_by_routing_tier() -> None:
    def model(routing_tier: str) -> ModelTier:
        return ModelTier(
            routing_tier=routing_tier,
            provider="hosted",
            model_name=f"{routing_tier}-model",
            deployment_name=f"{routing_tier}-deployment",
            role_tags=(),
            supports_tools=True,
            supports_json=True,
            long_context=True,
            coding_score=5.0,
            reasoning_score=5.0,
            latency_score=5.0,
            cost_score=5.0,
            reliability_score=5.0,
        )

    router = DeterministicRouter(
        models=(model("beta"), model("alpha")),
        profiles={"default": TaskProfile(capability="default")},
    )

    decision = router.route(RequestContext(capability="default"))

    assert [c.routing_tier for c in decision.ranked_candidates] == ["alpha", "beta"]


def test_invalid_policy_adjustment_key_is_rejected(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text(
        """
task_profiles:
  - capability: default
    priority_weight_adjustments:
      fastest:
        latency: 1.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown .*priority_weight_adjustments key"):
        load_task_profiles(profiles_path)
