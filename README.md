# ai-model-router

A lightweight, deterministic, capability-based model router.

- Selection only (no model execution)
- Filter-first, then score
- YAML-driven model and task profile config
- Deterministic ranked output with fallback chain and debug reasons

## Quick start

1. Edit [config/models.yaml](config/models.yaml)
2. Edit [config/task_profiles.yaml](config/task_profiles.yaml)
3. Instantiate `DeterministicRouter` and call `route(...)`

See [examples/usage_example.py](examples/usage_example.py).
