# EAF Cognitive Twin

EAF Cognitive Twin benchmark for industrial control and agentic AI policies.

## Benchmark simulator (Model C)
All policies are evaluated on the same enhanced hybrid first-principles simulator, Model C:
`Model_C_enhanced_hybrid = FirstPrinciplesModel(enhanced=True)`.

## Policies compared
- `baseline_schedule`
- `rule_based`
- `mpc`
- `agentic_ai`
- `q_learning`
- `dqn`
- `ppo`
- `behavior_cloning`
- `safe_ppo_agentic_mpc` (proposed)

## Proposed method
The proposed **Safe PPO-Agentic MPC** controller combines policy-gradient learning with model-based safety correction and rule-based tap gating. PPO provides adaptive control under nonlinear furnace dynamics, while MPC-style local lookahead corrects unsafe or inefficient actions before execution. A final safety filter enforces operational constraints.

## Reproducible commands
Training:
```bash
python -m agents.runners.train_agent \
  --config configs/base_case.json \
  --algorithm q_learning \
  --episodes 500 \
  --seed 7 \
  --output-dir results/agent_training/q_learning

python -m agents.runners.train_agent \
  --config configs/base_case.json \
  --algorithm dqn \
  --episodes 500 \
  --seed 7 \
  --output-dir results/agent_training/dqn

python -m agents.runners.train_agent \
  --config configs/base_case.json \
  --algorithm ppo \
  --episodes 1000 \
  --seed 7 \
  --output-dir results/agent_training/ppo

python -m agents.runners.train_agent \
  --config configs/base_case.json \
  --algorithm safe_ppo_agentic_mpc \
  --episodes 1000 \
  --seed 7 \
  --output-dir results/agent_training/safe_ppo_agentic_mpc
```

Evaluation:
```bash
python -m agents.runners.run_agent \
  --config configs/base_case.json \
  --output-dir results/agent_run \
  --seeds 30 \
  --n-scenarios 6 \
  --model C \
  --max-steps 650 \
  --mpc-horizon 8 \
  --include-rl-baselines \
  --report-format html,csv,md
```

Docker:
```bash
docker compose run --rm eaf-twin python -m agents.runners.run_agent \
  --config configs/base_case.json \
  --output-dir results/agent_run \
  --seeds 30 \
  --n-scenarios 6 \
  --model C \
  --max-steps 650 \
  --mpc-horizon 8 \
  --include-rl-baselines
```
