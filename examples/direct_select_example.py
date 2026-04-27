from pathlib import Path

from ai_model_selector.models import RequestContext
from ai_model_selector.selector import DeterministicSelector


root = Path(__file__).resolve().parents[1]
selector = DeterministicSelector.from_yaml(
    root / "config" / "models.yaml",
    root / "config" / "task_profiles.yaml",
)

context = RequestContext(
    capability="code.implement",
    needs_tools=True,
    needs_json=True,
    priority="quality",
)
decision = selector.select(context)

print("request context:", context)
print("primary tier:", decision.primary.selection_tier)
print("primary endpoint:", decision.primary)
print("fallback tiers:", list(decision.fallback_selection_tiers))
print("fallback endpoints:", list(decision.fallbacks))
