from pathlib import Path

from ai_model_router.config_loader import load_capability_definitions
from ai_model_router.intent import IntentResolver, build_request_context
from ai_model_router.router import DeterministicRouter


root = Path(__file__).resolve().parents[1]
router = DeterministicRouter.from_yaml(
    root / "config" / "models.yaml",
    root / "config" / "task_profiles.yaml",
)

resolver = IntentResolver(
    load_capability_definitions(root / "config" / "capabilities.yaml")
)

resolution = resolver.resolve("Design a scalable notification system")
context = build_request_context(resolution)
decision = router.route(context)

print("capability:", resolution.capability)
print("intent confidence:", resolution.confidence)
print("intent debug:", list(resolution.debug))
print("primary tier:", decision.primary.routing_tier)
print("primary endpoint:", decision.primary)
print("fallback tiers:", list(decision.fallback_routing_tiers))
print(
    "ranked:",
    [(c.routing_tier, c.model_name, c.score) for c in decision.ranked_candidates],
)
print(
    "score breakdown:",
    [(c.name, c.value, c.details) for c in decision.ranked_candidates[0].score_components],
)
print("debug:", list(decision.debug_reasons))
