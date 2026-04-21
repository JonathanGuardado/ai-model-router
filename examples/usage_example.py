from pathlib import Path

from ai_model_router.router import DeterministicRouter


root = Path(__file__).resolve().parents[1]
router = DeterministicRouter.from_yaml(
    root / "config" / "models.yaml",
    root / "config" / "task_profiles.yaml",
)

decision = router.route_prompt("hello")
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
