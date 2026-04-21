from pathlib import Path

from ai_model_router.models import RequestContext
from ai_model_router.router import DeterministicRouter


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "config" / "models.yaml"
PROFILES = ROOT / "config" / "task_profiles.yaml"


def _router() -> DeterministicRouter:
    return DeterministicRouter.from_yaml(MODELS, PROFILES)


def test_trivial_prefers_local_fast() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="trivial.classify",
            needs_json=True,
            budget_mode="economy",
        )
    )
    assert decision.primary_model == "local_fast"
    assert len(decision.fallback_models) == 2


def test_web_research_requires_tools_and_long_context() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="web.research",
            needs_tools=True,
            long_context=True,
        )
    )
    assert decision.primary_model == "web_agent"
    assert "local_fast" not in [c.model_name for c in decision.ranked_candidates]


def test_code_implement_prefers_coding_primary() -> None:
    router = _router()
    decision = router.route(
        RequestContext(
            capability="code.implement",
            needs_tools=True,
            needs_json=True,
            priority="quality",
        )
    )
    assert decision.primary_model == "coding_primary"


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
    assert decision.primary_model == "reasoning_primary"


def test_deterministic_same_input_same_output() -> None:
    router = _router()
    context = RequestContext(
        capability="tests.analyze",
        needs_json=True,
        retry_count=1,
    )
    first = router.route(context)
    second = router.route(context)

    assert first.primary_model == second.primary_model
    assert first.fallback_models == second.fallback_models
    assert [c.model_name for c in first.ranked_candidates] == [
        c.model_name for c in second.ranked_candidates
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
    assert decision.primary_model == "local_fast"
