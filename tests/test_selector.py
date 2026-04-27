from pathlib import Path

import pytest

from ai_model_selector.config_loader import load_model_tiers, load_task_profiles
from ai_model_selector.models import (
    ModelSelection,
    ModelTier,
    RequestContext,
    RequiredConstraints,
    TaskProfile,
)
from ai_model_selector.selector import DeterministicSelector, SelectionError


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "config" / "models.yaml"
PROFILES = ROOT / "config" / "task_profiles.yaml"
CAPABILITIES = ROOT / "config" / "capabilities.yaml"


def _selector() -> DeterministicSelector:
    return DeterministicSelector.from_yaml(MODELS, PROFILES)


def _selector_with_prompt_helper() -> DeterministicSelector:
    return DeterministicSelector.from_yaml(MODELS, PROFILES, CAPABILITIES)


def test_select_with_explicit_trivial_capability_prefers_local_fast() -> None:
    selector = _selector()
    decision = selector.select(
        RequestContext(
            capability="trivial.classify",
            needs_json=True,
            budget_mode="economy",
        )
    )
    assert decision.primary_selection_tier == "local_fast"
    assert decision.primary_model == "local_model"
    assert decision.primary.provider == "local"
    assert decision.primary.invocation == "openai_chat"
    assert len(decision.fallback_models) == 2


def test_select_prompt_convenience_uses_intent_resolver_for_hello() -> None:
    selector = _selector_with_prompt_helper()

    decision = selector.select_prompt("hello")

    assert decision.capability == "trivial.respond"
    assert decision.primary_selection_tier == "local_fast"
    assert decision.primary_model == "local_model"


def test_select_prompt_convenience_uses_intent_resolver_for_code_request() -> None:
    selector = _selector_with_prompt_helper()

    decision = selector.select_prompt("Write a Python function and tests for sorting users")

    assert decision.capability == "code.implement"
    assert decision.primary_selection_tier == "coding_primary"


def test_select_prompt_requires_configured_capabilities() -> None:
    selector = _selector()

    with pytest.raises(SelectionError, match="Pass capabilities_path to from_yaml"):
        selector.select_prompt("Write a Python function and tests")


def test_models_yaml_without_invocation_defaults_to_openai_chat(tmp_path: Path) -> None:
    models_path = tmp_path / "models.yaml"
    models_path.write_text(
        """
models:
  - selection_tier: local_fast
    provider: local
    model_name: local_model
    deployment_name: local_model
    role_tags: []
    supports_tools: false
    supports_json: true
    long_context: false
    coding_score: 1
    reasoning_score: 1
    latency_score: 1
    cost_score: 1
    reliability_score: 1
""",
        encoding="utf-8",
    )

    models = load_model_tiers(models_path)

    assert models[0].invocation == "openai_chat"


def test_select_with_explicit_web_research_capability_requires_tools() -> None:
    selector = _selector()
    decision = selector.select(
        RequestContext(
            capability="web.research",
            needs_tools=True,
            long_context=True,
        )
    )
    assert decision.primary_selection_tier == "web_agent"
    assert "local_fast" not in [c.selection_tier for c in decision.ranked_candidates]
    assert any(c.selection_tier == "local_fast" for c in decision.filtered_candidates)


def test_select_with_explicit_code_capability_prefers_coding_primary() -> None:
    selector = _selector()
    decision = selector.select(
        RequestContext(
            capability="code.implement",
            needs_tools=True,
            needs_json=True,
            priority="quality",
        )
    )
    assert decision.primary_selection_tier == "coding_primary"
    assert decision.primary.deployment_name == "coding_model"
    assert decision.primary.invocation == "openai_chat"


def test_selection_plan_preserves_endpoint_metadata_from_models_yaml(
    tmp_path: Path,
) -> None:
    models_path = tmp_path / "models.yaml"
    profiles_path = tmp_path / "profiles.yaml"
    models_path.write_text(
        """
models:
  - selection_tier: coding_primary
    provider: openrouter
    model_name: descriptive-coder
    deployment_name: provider/coder-runtime-id
    invocation: openai_chat
    role_tags: [coding]
    supports_tools: true
    supports_json: true
    long_context: true
    coding_score: 10
    reasoning_score: 7
    latency_score: 5
    cost_score: 5
    reliability_score: 8
  - selection_tier: local_fast
    provider: local
    model_name: local-fast-label
    deployment_name: local-runtime-id
    invocation: openai_chat
    role_tags: [fast]
    supports_tools: true
    supports_json: true
    long_context: true
    coding_score: 5
    reasoning_score: 4
    latency_score: 10
    cost_score: 10
    reliability_score: 6
""",
        encoding="utf-8",
    )
    profiles_path.write_text(
        """
task_profiles:
  - capability: code.implement
    preferred_tiers: [coding_primary]
    scoring_weights:
      coding: 1
    fallback_count: 1
""",
        encoding="utf-8",
    )
    selector = DeterministicSelector.from_yaml(models_path, profiles_path)

    decision = selector.select(RequestContext(capability="code.implement"))

    assert decision.primary == ModelSelection(
        selection_tier="coding_primary",
        provider="openrouter",
        model_name="descriptive-coder",
        deployment_name="provider/coder-runtime-id",
        invocation="openai_chat",
    )
    assert decision.fallbacks == (
        ModelSelection(
            selection_tier="local_fast",
            provider="local",
            model_name="local-fast-label",
            deployment_name="local-runtime-id",
            invocation="openai_chat",
        ),
    )
    assert decision.ranked_candidates[0].deployment_name == "provider/coder-runtime-id"
    assert decision.ranked_candidates[0].invocation == "openai_chat"


def test_fallbacks_skip_duplicate_execution_endpoints(tmp_path: Path) -> None:
    models_path = tmp_path / "models.yaml"
    profiles_path = tmp_path / "profiles.yaml"
    models_path.write_text(
        """
models:
  - selection_tier: coding_primary
    provider: minimax
    model_name: MiniMax-M2.7
    deployment_name: MiniMax-M2.7
    invocation: openai_chat
    role_tags: [coding]
    supports_tools: true
    supports_json: true
    long_context: true
    coding_score: 10
    reasoning_score: 8
    latency_score: 5
    cost_score: 5
    reliability_score: 8
  - selection_tier: design_primary
    provider: minimax
    model_name: MiniMax-M2.7
    deployment_name: MiniMax-M2.7
    invocation: openai_chat
    role_tags: [architecture]
    supports_tools: true
    supports_json: true
    long_context: true
    coding_score: 9
    reasoning_score: 9
    latency_score: 5
    cost_score: 5
    reliability_score: 8
  - selection_tier: web_fallback
    provider: gemini
    model_name: gemini-flash
    deployment_name: gemini-2.5-flash
    invocation: openai_chat
    role_tags: [fallback]
    supports_tools: true
    supports_json: true
    long_context: true
    coding_score: 6
    reasoning_score: 6
    latency_score: 10
    cost_score: 8
    reliability_score: 8
""",
        encoding="utf-8",
    )
    profiles_path.write_text(
        """
task_profiles:
  - capability: code.implement
    scoring_weights:
      coding: 1
    fallback_count: 2
""",
        encoding="utf-8",
    )
    selector = DeterministicSelector.from_yaml(models_path, profiles_path)

    decision = selector.select(RequestContext(capability="code.implement"))

    assert decision.primary.selection_tier == "coding_primary"
    assert decision.fallback_selection_tiers == ("web_fallback",)
    assert [candidate.selection_tier for candidate in decision.ranked_candidates] == [
        "coding_primary",
        "design_primary",
        "web_fallback",
    ]
    assert any("deduped_endpoint:design_primary" in item for item in decision.debug_reasons)


def test_fallbacks_are_full_endpoint_objects_and_compat_properties() -> None:
    selector = _selector()

    decision = selector.select(
        RequestContext(
            capability="code.implement",
            needs_tools=True,
            needs_json=True,
            priority="quality",
        )
    )

    assert all(isinstance(fallback, ModelSelection) for fallback in decision.fallbacks)
    assert all(fallback.provider for fallback in decision.fallbacks)
    assert all(fallback.deployment_name for fallback in decision.fallbacks)
    assert decision.fallback_models == tuple(
        fallback.model_name for fallback in decision.fallbacks
    )
    assert decision.fallback_selection_tiers == tuple(
        fallback.selection_tier for fallback in decision.fallbacks
    )


def test_architecture_design_retry_escalates_to_reasoning_primary() -> None:
    selector = _selector()
    decision = selector.select(
        RequestContext(
            capability="architecture.design",
            retry_count=1,
            long_context=True,
            priority="quality",
            budget_mode="premium",
        )
    )
    assert decision.primary_selection_tier == "reasoning_primary"


def test_deterministic_same_input_same_output() -> None:
    selector = _selector()
    context = RequestContext(
        capability="tests.analyze",
        needs_json=True,
        retry_count=1,
    )
    first = selector.select(context)
    second = selector.select(context)

    assert first.primary_selection_tier == second.primary_selection_tier
    assert first.fallback_selection_tiers == second.fallback_selection_tiers
    assert [c.selection_tier for c in first.ranked_candidates] == [
        c.selection_tier for c in second.ranked_candidates
    ]


def test_budget_economy_in_default_profile_prefers_local_fast() -> None:
    selector = _selector()
    decision = selector.select(
        RequestContext(
            capability="unknown.capability",
            budget_mode="economy",
            priority="latency",
        )
    )
    assert decision.primary_selection_tier == "local_fast"
    assert decision.profile == "default"


def test_score_breakdown_includes_configured_policy_adjustments() -> None:
    selector = _selector()
    decision = selector.select(
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


def test_select_raises_when_no_compatible_model() -> None:
    selector = DeterministicSelector(
        models=(
            ModelTier(
                selection_tier="plain_text_only",
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

    with pytest.raises(SelectionError):
        selector.select(
            RequestContext(
                capability="web.research",
                needs_tools=True,
                long_context=True,
                needs_json=True,
            )
        )


def test_deterministic_tie_breaks_by_selection_tier() -> None:
    def model(selection_tier: str) -> ModelTier:
        return ModelTier(
            selection_tier=selection_tier,
            provider="hosted",
            model_name=f"{selection_tier}-model",
            deployment_name=f"{selection_tier}-deployment",
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

    selector = DeterministicSelector(
        models=(model("beta"), model("alpha")),
        profiles={"default": TaskProfile(capability="default")},
    )

    decision = selector.select(RequestContext(capability="default"))

    assert [c.selection_tier for c in decision.ranked_candidates] == ["alpha", "beta"]


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
