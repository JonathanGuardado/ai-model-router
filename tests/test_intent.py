from pathlib import Path

from ai_model_selector.config_loader import load_capability_definitions
from ai_model_selector.intent import IntentResolver, build_request_context
from ai_model_selector.selector import DeterministicSelector

ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES = ROOT / "config" / "capabilities.yaml"
MODELS = ROOT / "config" / "models.yaml"
PROFILES = ROOT / "config" / "task_profiles.yaml"


def _resolver() -> IntentResolver:
    return IntentResolver(load_capability_definitions(CAPABILITIES))


def test_resolves_greeting_to_trivial_respond() -> None:
    resolution = _resolver().resolve("hello")

    assert resolution.capability == "trivial.respond"
    assert resolution.ambiguous is False
    assert resolution.signals["needs_json"] is True


def test_resolves_web_research_request() -> None:
    resolution = _resolver().resolve(
        "Research the latest pricing for hosted GPU inference and include sources."
    )

    assert resolution.capability == "web.research"
    assert resolution.signals["needs_tools"] is True
    assert any("time_sensitive_language" in item for item in resolution.debug)


def test_resolves_coding_request() -> None:
    resolution = _resolver().resolve(
        "Implement a Python function and tests for parsing user records."
    )

    assert resolution.capability == "code.implement"
    assert resolution.signals["needs_tools"] is True
    assert resolution.signals["needs_json"] is True


def test_resolves_architecture_request() -> None:
    resolution = _resolver().resolve(
        "Design a scalable event processing architecture and explain tradeoffs."
    )

    assert resolution.capability == "architecture.design"
    assert resolution.signals["budget_mode"] == "premium"
    assert any("architecture_language" in item for item in resolution.debug)


def test_weak_input_is_ambiguous_and_uses_safe_default() -> None:
    resolution = _resolver().resolve("help with this")

    assert resolution.ambiguous is True
    assert resolution.capability == "trivial.respond"
    assert resolution.source in {"ambiguous", "default"}
    assert len(resolution.candidates) >= 2


def test_builds_request_context_from_resolution() -> None:
    resolution = _resolver().resolve(
        "Research the latest Python release notes and summarize them."
    )

    context = build_request_context(resolution, retry_count=2)

    assert context.capability == "web.research"
    assert context.retry_count == 2
    assert context.needs_tools is True
    assert context.long_context is True
    assert context.priority == "quality"


def test_end_to_end_resolve_build_context_then_select() -> None:
    resolver = _resolver()
    selector = DeterministicSelector.from_yaml(MODELS, PROFILES)
    resolution = resolver.resolve("Write a Python function and tests for sorting users.")
    context = build_request_context(resolution)

    decision = selector.select(context)

    assert resolution.capability == "code.implement"
    assert decision.primary_selection_tier == "coding_primary"
    assert decision.primary.provider == "hosted"
    assert decision.primary.model_name == "coding_model"
    assert decision.primary.deployment_name == "coding_model"
    assert decision.primary.invocation == "openai_chat"
    assert all(fallback.deployment_name for fallback in decision.fallbacks)
