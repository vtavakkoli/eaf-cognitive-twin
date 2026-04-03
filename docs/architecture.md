# Architecture

The project is organized as a layered simulation stack:

- `config/`: defaults, scenario generation, schema checks.
- `domain/`: core entities and state dataclasses.
- `models/`: model A/B/C implementations.
- `simulation/`: scheduler + orchestration.
- `validation/`: physical guardrails and plausibility checks.
- `reporting/`: plotting and summary production.
- `estimation/`: calibration scaffold.
- `io/`: CSV/JSON persistence.

`cli.py` is intentionally thin and delegates to simulation services.
