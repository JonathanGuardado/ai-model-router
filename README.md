# ai-model-selector

Deterministic intent resolution and model selection for AI systems.

## What It Does

- Resolves free-form text into an internal capability.
- Converts that capability into a `RequestContext`.
- Selects a primary model endpoint and fallback endpoints.
- Returns a stable execution plan for downstream routers.

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

`select_prompt(...)` is available as a convenience helper when the selector was constructed with capabilities:

```python
selector = DeterministicSelector.from_yaml(
    "config/models.yaml",
    "config/task_profiles.yaml",
    capabilities_path="config/capabilities.yaml",
)

decision = selector.select_prompt("Design a scalable notification system")
```

Calling `select_prompt(...)` without configured capabilities raises a `SelectionError`. The explicit flow above is the recommended integration path for apps that already own intent resolution.

## App/Router Contract

The selector answers: **which model endpoint should handle this request?**

The app/router answers: **how do I call this provider and deployment?**

`SelectionDecision.primary` and every item in `SelectionDecision.fallbacks` are `ModelSelection` endpoint objects:

```python
selection.primary.selection_tier   # stable app-facing policy/interface name
selection.primary.provider         # provider/client lookup key
selection.primary.model_name       # descriptive model identity
selection.primary.deployment_name  # exact model/deployment string to send
selection.primary.invocation       # call style, defaults to "openai_chat"
```

Routers should send `endpoint.deployment_name` as the provider payload model value and preserve `endpoint.selection_tier` in metadata/logs. They should not keep a separate `selection_tier -> model` mapping, because model policy belongs in this package.

Compatibility properties are still available:

```python
decision.primary_model
decision.primary_selection_tier
decision.fallback_models
decision.fallback_selection_tiers
```

## Example Capabilities

- `trivial.respond`
- `web.research`
- `code.implement`
- `code.verify`
- `architecture.design`

## What It Does Not Do

- Does not execute model calls.
- Does not call providers or remote APIs.
- Does not know provider credentials or secrets.
- Does not orchestrate workflows or agents.
- Does not include runtime or tool execution integrations.

## Config

Configuration is defined in `config/`:

- `capabilities.yaml`: intent categories and default request signals
- `models.yaml`: available model tiers and endpoint metadata, including `provider`, `deployment_name`, and optional `invocation`
- `task_profiles.yaml`: selection policy per capability
