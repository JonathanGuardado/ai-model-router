from pathlib import Path

from ai_model_router.models import RequestContext
from ai_model_router.router import DeterministicRouter


root = Path(__file__).resolve().parents[1]
router = DeterministicRouter.from_yaml(
    root / "config" / "models.yaml",
    root / "config" / "task_profiles.yaml",
)

context = RequestContext(
    capability="code.implement",
    needs_tools=True,
    needs_json=True,
    priority="quality",
)
decision = router.route(context)

print("capability:", context.capability)
print("primary tier:", decision.primary.routing_tier)
print("primary endpoint:", decision.primary)
print("fallback tiers:", list(decision.fallback_routing_tiers))
