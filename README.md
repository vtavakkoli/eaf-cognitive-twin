# EAF Cognitive Twin

EAF Cognitive Twin for comparing industrial control policies under an enhanced hybrid simulator.

## Benchmark simulator (Model C)
Model C is the enhanced hybrid first-principles simulator used as the benchmark environment:
`Model_C_enhanced_hybrid = FirstPrinciplesModel(enhanced=True)`.

All benchmark policies are evaluated on the same Model C simulator for fair comparison.

## Policies
- `baseline_schedule`
- `rule_based`
- `mpc`
- `agentic_ai`

## Scenarios
- `base_case`
- `higher_oxygen`
- `higher_natural_gas`
- `improved_foamy_slag`
- `dri20`
- `delayed_melting_downtime`

## Metrics
- reward
- tap success
- tapped mass
- energy
- oxygen
- natural gas
- temperature violations
- feasibility

## Reproducible commands
```bash
python -m agents.runners.train_agent \
  --config configs/base_case.json \
  --output-dir results/agent_training \
  --iterations 100 \
  --seed 7

python -m agents.runners.run_agent \
  --config configs/base_case.json \
  --output-dir results/agent_run \
  --trained-policy results/agent_training/checkpoints/best_policy.json \
  --seeds 10 \
  --model C \
  --mpc-horizon 8
```

## Expected outputs
- `results/agent_run/result.html`
- `results/agent_run/scenario_summary.csv`
- `results/agent_run/policy_aggregate_summary.csv`
- `results/agent_run/statistical_analysis.csv`
- `results/agent_run/timeseries/*.csv`
- `results/agent_run/figures/*.png`

## Tests
```bash
python -m pytest tests
```
