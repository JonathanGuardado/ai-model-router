# ai-model-selector

Deterministic intent resolution and model selection for agentic AI systems.

## What It Does

- Resolves free-form text into an internal capability.
- Converts that capability into a `RequestContext`.
- Selects a primary model tier and fallback tiers.
- Returns selection metadata for downstream routing or execution.

## Main Flow

```text
IntentResolver.resolve(...)
-> build_request_context(...)
-> DeterministicSelector.select(...)
```

```python
from ai_model_selector.config_loader import load_capability_definitions
from ai_model_selector.intent import IntentResolver, build_request_context
from ai_model_selector.selector import DeterministicSelector

resolver = IntentResolver(load_capability_definitions("config/capabilities.yaml"))
selector = DeterministicSelector.from_yaml(
    "config/models.yaml",
    "config/task_profiles.yaml",
)

resolution = resolver.resolve("Design a scalable notification system")
context = build_request_context(resolution)
decision = selector.select(context)

print(resolution.capability)
print(decision.primary)
```

`select_prompt(...)` is available as a convenience helper, but the explicit flow above is the recommended integration path.

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
- Does not include runtime or tool execution integrations.

## Config

Configuration is defined in `config/`:

- `capabilities.yaml`: intent categories and default request signals
- `models.yaml`: available model tiers and endpoint metadata
- `task_profiles.yaml`: selection policy per capability
