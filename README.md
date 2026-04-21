# ai-model-router

A lightweight, deterministic, capability-based model router.

- Selection only (no model execution)
- Filter-first, then score
- YAML-driven model, task profile, and routing policy config
- Separate routing tiers from provider model/deployment endpoints
- Deterministic ranked output with fallback chain and structured score/debug reasons

## Quick start

1. Edit [config/models.yaml](config/models.yaml)
2. Edit [config/task_profiles.yaml](config/task_profiles.yaml)
3. Instantiate `DeterministicRouter` and call `route_prompt(...)`

See [examples/usage_example.py](examples/usage_example.py).

```python
decision = router.route_prompt("hello")
print(decision.primary)
```

For advanced use, build a `RequestContext` yourself and call `route(...)`.

## Config shape

`models.yaml` describes available endpoint candidates. `routing_tier` is the stable
policy-facing name used by profiles, while `provider`, `model_name`, and
`deployment_name` describe the integration endpoint.

`task_profiles.yaml` describes routing policy per capability: required constraints,
base scoring weights, priority/budget weight adjustments, boosts, penalties,
retry escalation, and fallback count.

## Result shape

`route(...)` returns a `RoutingDecision` with:

- `primary`: selected `ModelSelection` endpoint details
- `fallbacks`: ordered fallback endpoint details
- `ranked_candidates`: all compatible candidates with score components
- `filtered_candidates`: candidates excluded during compatibility filtering
- `debug_reasons`: compact string reasons for logs
