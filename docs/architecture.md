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

## Goal-Conditioned JEPA-PPO-TD3BC

The goal-conditioned JEPA-PPO policy uses TD3+BC as a goal setter instead of
using fixed recipe targets only. The controller receives the normalized furnace
state, previous action, recipe set-point, TD3BC short-horizon goal, goal error,
and process phase. The JEPA module predicts the next latent furnace state, and
the PPO head maps this enriched latent representation to a safe discrete EAF
action.

```text
z_t, u_{t-1}, r_t, g_t^{TD3BC}, e_t, p_t
        -> JEPA predictor \hat{z}_{t+1}
        -> PPO policy pi(a_t | z_t, \hat{z}_{t+1}, g_t^{TD3BC})
        -> simulator safety filter
```

Here, `g_t^{TD3BC}` is built by blending a TD3-inspired continuous target with
a behavior-cloning expert prior. This reflects the benchmark finding that
`PPO-SafeAgent-TD3BC` and `behavior_cloning` are strong baselines: they are now
used to shape the goal embedding, while PPO remains responsible for the final
adaptive decision.
