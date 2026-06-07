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

## TD3BC-Guided JEPA-PPO SafeAgent

The goal-conditioned JEPA policy is now a residual improvement layer on top of
`PPO-SafeAgent-TD3BC`, not an independent replacement policy. This change fixes
the observed failure mode where the standalone JEPA-PPO actor had good endpoint
quality but missed tap-ready states.

The controller receives the normalized furnace state, previous action, recipe
set-point, TD3BC short-horizon goal, goal error, and process phase. The JEPA
module predicts the next latent furnace state. That prediction is then used to
apply a conservative residual correction around the `PPO-SafeAgent-TD3BC`
execution backbone.

```text
z_t, u_{t-1}, r_t, g_t^{TD3BC}, e_t, p_t
        -> JEPA predictor \hat{z}_{t+1}
        -> residual tap-ready / endpoint-quality shaping
        -> PPO-SafeAgent-TD3BC execution backbone
        -> simulator safety filter
```

Here, `g_t^{TD3BC}` is built by blending a TD3-inspired continuous target with
a behavior-cloning expert prior. JEPA now improves the strongest baseline by
recovering late/cold heats, protecting melt trajectory, and tapering endpoint
overshoot, while the safe TD3BC backbone keeps the action envelope stable.
