# EAF Cognitive Digital Twin

Refactored research-grade foundation for an Electric Arc Furnace (EAF) cognitive digital twin with three fidelity levels:
1. empirical
2. first-principles
3. enhanced hybrid

This repository now also includes a top-level `agents/` package that implements an external control layer over `eaf_twin` for training and benchmark runs.

## Quick start (local)
```bash
pip install -r requirements.txt
pip install -e .
python -m eaf_twin.cli run --config configs/base_case.json --output-dir outputs
```

## Agent commands (Python)
Train/tune a trainable external control policy:
```bash
python -m agents.runners.train_agent --config configs/base_case.json --output-dir results/agent_training
```

Run benchmark scenarios with baseline schedule, rule-based policy, and trained policy (if available):
```bash
python -m agents.runners.run_agent --config configs/base_case.json --output-dir results/agent_run --trained-policy results/agent_training/checkpoints/best_policy.json
```
This run compares four control strategies when a trained checkpoint exists: `baseline_schedule`, `rule_based`, `mpc`, and `agentic_ai`.

## Docker Compose commands
```bash
docker compose up --build full-run
docker compose up --build train-agent
docker compose up --build run-agent
```

## Outputs
### Full simulation output
- `outputs/summary_all_scenarios.csv`
- `outputs/summary_all_scenarios.json`
- `outputs/timeseries_<scenario>_<model>.csv`
- `outputs/plot_*.png`

### Agent training output
- `results/agent_training/training_log.csv`
- `results/agent_training/checkpoints/*.json`
- `results/agent_training/training_summary.json`
- `results/agent_training/training_reward_curve.png`

### Agent run output
- `results/agent_run/scenario_summary.csv`
- `results/agent_run/policy_aggregate_summary.csv`
- `results/agent_run/kpi_comparison.csv`
- `results/agent_run/statistical_analysis.csv`
- `results/agent_run/timeseries/agent_timeseries_<scenario>_<policy>.csv`
- `results/agent_run/plot_*_comparison.png`
- `results/agent_run/plot_composite_policy_score.png`
- `results/agent_run/plot_reward_heatmap.png`
- `results/agent_run/report.md`
- `results/agent_run/result.html`

## Inspecting results
- Use `scenario_summary.csv` and `kpi_comparison.csv` for scenario-level comparisons.
- Use per-scenario timeseries CSVs to inspect action traces and state trajectories.
- Use generated PNG charts for quick visual comparison of energy, consumables, temperatures, and outcome KPIs.

## Tests
```bash
python -m unittest discover -s tests -v
```
