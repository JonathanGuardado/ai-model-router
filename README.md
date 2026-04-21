# ai-model-router

A lightweight, deterministic intent and model routing package.

- Selection only (no model execution)
- Intent resolution converts free-form text into internal capabilities
- Model routing converts structured capabilities into model selections
- Router remains filter-first, then score
- YAML-driven capability, model, task profile, and routing policy config
- Separate routing tiers from provider model/deployment endpoints
- Deterministic ranked output with fallback chain and structured score/debug reasons

## Two Layers

`IntentResolver` answers: "What kind of task is this text asking for?"

Input: free-form text

Output: `CapabilityResolution`

`DeterministicRouter` answers: "Which configured model tier should handle this task?"

Input: `RequestContext`

Output: `RoutingDecision`

The intended flow is:

```text
free-form text
  -> IntentResolver.resolve(...)
  -> CapabilityResolution
  -> build_request_context(...)
  -> DeterministicRouter.route(...)
  -> RoutingDecision
```

No remote calls or LLM classification calls are used in the normal path.

## Quick start

1. Edit [config/capabilities.yaml](config/capabilities.yaml)
2. Edit [config/models.yaml](config/models.yaml)
3. Edit [config/task_profiles.yaml](config/task_profiles.yaml)
4. Resolve intent, build a `RequestContext`, and route it

See [examples/usage_example.py](examples/usage_example.py).

```python
from ai_model_router.config_loader import load_capability_definitions
from ai_model_router.intent import IntentResolver, build_request_context
from ai_model_router.router import DeterministicRouter

resolver = IntentResolver(load_capability_definitions("config/capabilities.yaml"))
router = DeterministicRouter.from_yaml(
    "config/models.yaml",
    "config/task_profiles.yaml",
)

resolution = resolver.resolve("hello")
context = build_request_context(resolution)
decision = router.route(context)

print(resolution.capability)
print(decision.primary)
```

For compatibility, `DeterministicRouter.route_prompt(...)` is available when the
router is constructed with a configured intent resolver via `capabilities_path`.

## Config shape

`capabilities.yaml` describes intent categories. Each capability has a name,
description, examples, anti-examples, and default request signals.

`models.yaml` describes available endpoint candidates. `routing_tier` is the stable
policy-facing name used by profiles, while `provider`, `model_name`, and
`deployment_name` describe the integration endpoint.

`task_profiles.yaml` describes routing policy per capability: required constraints,
base scoring weights, priority/budget weight adjustments, boosts, penalties,
retry escalation, and fallback count.

## Intent Resolution

Intent resolution is deterministic and layered:

- Semantic capability matching compares input text against configured capability descriptions, examples, and anti-examples using a local matcher.
- Structural heuristics apply small explainable score adjustments for signals like greetings, URLs, code blocks, stack traces, file paths, implementation language, and architecture language.
- Ambiguity handling marks weak or close results as ambiguous, returns top candidates with confidence, and falls back safely without remote calls.

The matcher is pluggable, so the default local matcher can later be replaced with
a local embedding matcher or classifier without changing the router.

## Result shape

`route(...)` returns a `RoutingDecision` with:

- `primary`: selected `ModelSelection` endpoint details
- `fallbacks`: ordered fallback endpoint details
- `ranked_candidates`: all compatible candidates with score components
- `filtered_candidates`: candidates excluded during compatibility filtering
- `debug_reasons`: compact string reasons for logs

`IntentResolver.resolve(...)` returns a `CapabilityResolution` with:

- `capability`: selected internal capability
- `confidence`: deterministic confidence derived from candidate separation
- `source`: `semantic`, `ambiguous`, or `default`
- `ambiguous`: whether the result is weak or close
- `signals`: default and structural signals used to build `RequestContext`
- `candidates`: ranked capability candidates
- `debug`: explainable resolution details
