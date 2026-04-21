# ai-model-router

Deterministic intent resolution and model routing for selection-only AI systems.

## What It Does

- Resolves free-form text into an internal capability.
- Converts that capability into a `RequestContext`.
- Selects a primary model tier and fallback tiers.
- Returns routing metadata without executing any model.

## Main Flow

```text
IntentResolver.resolve(...)
-> build_request_context(...)
-> DeterministicRouter.route(...)
```

```python
from ai_model_router.config_loader import load_capability_definitions
from ai_model_router.intent import IntentResolver, build_request_context
from ai_model_router.router import DeterministicRouter

resolver = IntentResolver(load_capability_definitions("config/capabilities.yaml"))
router = DeterministicRouter.from_yaml(
    "config/models.yaml",
    "config/task_profiles.yaml",
)

resolution = resolver.resolve("Design a scalable notification system")
context = build_request_context(resolution)
decision = router.route(context)

print(resolution.capability)
print(decision.primary)
```

`route_prompt(...)` exists as a convenience helper, but the explicit flow above is
the recommended path.

## Example Capabilities

- `trivial.respond`
- `web.research`
- `code.implement`
- `code.verify`
- `architecture.design`

## What It Does Not Do

- Does not execute model calls.
- Does not call providers or remote APIs.
- Does not orchestrate workflows or agents.
- Does not integrate with Slack, Jira, LangGraph, OpenRouter, or LiteLLM.

## Config

Configuration lives in `config/`:

- `capabilities.yaml`: intent categories and default request signals
- `models.yaml`: available model tiers and endpoint metadata
- `task_profiles.yaml`: routing policy per capability
