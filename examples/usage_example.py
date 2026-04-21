from pathlib import Path

from ai_model_router.models import RequestContext
from ai_model_router.router import DeterministicRouter


root = Path(__file__).resolve().parents[1]
router = DeterministicRouter.from_yaml(
    root / "config" / "models.yaml",
    root / "config" / "task_profiles.yaml",
)

ctx = RequestContext(
    capability="code.implement",
    retry_count=0,
    needs_tools=True,
    needs_json=True,
    long_context=False,
    priority="quality",
    budget_mode="balanced",
)

decision = router.route(ctx)
print("primary:", decision.primary_model)
print("fallbacks:", list(decision.fallback_models))
print("ranked:", [(c.model_name, c.score) for c in decision.ranked_candidates])
print("debug:", list(decision.debug_reasons))
