# EAF Cognitive Twin

EAF Cognitive Twin is a **multi-fidelity digital-twin benchmark** for electric arc furnace (EAF) steelmaking, designed for reproducible evaluation of industrial control and agentic AI policies under shared simulation conditions.

## Digital Twin Model Hierarchy

This repository contains three EAF simulator variants with increasing realism and process fidelity.

1. **Model A – Empirical / baseline model**
   - **File:** `src/eaf_twin/models/empirical.py`
   - **Class/name:** `EmpiricalModel` / `Model_A_empirical`
   - **Purpose:** lightweight reference model for fast sanity checks, simple experiments, and baseline comparisons.
   - **Interpretation:** Model A is useful for reproducibility and quick testing, but it does not represent the full thermo-metallurgical furnace dynamics.

2. **Model B – First-principles model**
   - **File:** `src/eaf_twin/models/first_principles.py`
   - **Class/name:** `FirstPrinciplesModel` / `Model_B_first_principles`
   - **Purpose:** physics-based EAF simulator using energy, mass, heat-transfer, melting, slag, off-gas, chemical-energy, and tapping-related dynamics.
   - **Interpretation:** Model B is more interpretable and physically grounded than Model A.

3. **Model C – Enhanced hybrid digital twin**
   - **File:** `src/eaf_twin/models/first_principles.py`
   - **Created by:** `FirstPrinciplesModel(enhanced=True)`
   - **Runtime name:** `Model_C_enhanced_hybrid`
   - **Purpose:** most realistic benchmark model in this repository.
   - **Interpretation:** Model C extends the first-principles formulation with enhanced hybrid behaviour, richer process dynamics, improved phase/melting behaviour, chemical-energy interactions, foamy-slag/arc-efficiency effects, tapping logic, operational constraints, and sensor noise.
   - **Benchmark guidance:** all agentic control and policy benchmark results should use Model C unless a specific ablation compares A/B/C.
   - **Research role:** Model C is the preferred research simulator for future EAF digital-twin studies, safe RL, agentic AI control, policy benchmarking, and reproducible comparison of industrial control strategies.

| Model | Implementation | Fidelity | Main Use | Recommended For |
|---|---|---|---|---|
| Model A | `empirical.py` | Low fidelity | Fast baseline | Sanity checks and quick tests |
| Model B | `first_principles.py` | Medium/high fidelity | Physics-based simulation | Interpretable ablations and physics validation |
| Model C | `first_principles.py` with `enhanced=True` | Highest fidelity in this repository | Enhanced hybrid digital twin | Main benchmark and future EAF research |

## Benchmark Simulator: Model C Enhanced Hybrid Digital Twin

The benchmark intentionally evaluates policies on one common simulator, **Model C**, to ensure fair comparison across methods under identical closed-loop conditions. In the agentic benchmark, Model C is selected because it provides a more realistic closed-loop environment than Model A and Model B while remaining suitable for reproducible benchmarking and controlled analysis.

`Model_C_enhanced_hybrid = FirstPrinciplesModel(enhanced=True)`.

### Why Model C is important for later research

- It provides a reusable EAF digital-twin environment for future safe RL and agentic AI research.
- It enables controlled comparison of rule-based, MPC, RL, imitation-learning, and hybrid SafeAgent policies.
- It supports reproducible benchmarking through common scenarios, seeds, metrics, and reports.
- It can be extended later with plant calibration data, uncertainty estimation, online adaptation, and human-in-the-loop operator validation.
- It should be described as a research-grade simulator, not as a certified plant controller.

```text
Industrial EAF process assumptions
        ↓
Model A / Model B / Model C simulators
        ↓
Scenario generation and controlled episodes
        ↓
Baseline, MPC, RL, BC, and SafeAgent policies
        ↓
Benchmark metrics, reports, and future research extensions
```

The goal-conditioned JEPA-PPO variant now uses **TD3+BC as the goal setter**:

```text
current multimodal EAF state
        ↓
TD3 smooth target + Behavior Cloning expert prior
        ↓
short-horizon TD3BC goal embedding
        ↓
JEPA latent next-state predictor
        ↓
PPO policy head + simulator safety filter
        ↓
safe EAF control action
```

## Policies compared
- `baseline_schedule`
- `rule_based`
- `mpc`
- `trainable_adaptive_controller`
- `q_learning`
- `dqn`
- `ppo`
- `goal_conditioned_jepa_ppo` *(Goal-Conditioned JEPA-PPO-TD3BC)*
- `behavior_cloning`
- `sac_td3_inspired_heuristics` *(if available in the repository)*
- `safe_ppo_agentic_mpc`
- `safe_ppo_agentic_hybrid_variants` *(if available in the repository)*

## Proposed method
One proposed method in this benchmark is **Safe PPO-Agentic TD3 BC**, which combines policy-gradient learning with model-based safety correction and rule-based tap gating. PPO provides adaptive control under nonlinear furnace dynamics, while TD3 and BC local lookahead corrects unsafe or inefficient actions before execution. A final safety filter enforces operational constraints.

The **Goal-Conditioned JEPA-PPO-TD3BC** model reuses this strongest prior differently: TD3+BC does not directly replace PPO. Instead, it produces the intermediate goal embedding used by the JEPA predictor and PPO policy. This keeps PPO as the final adaptive policy while letting TD3+BC define realistic short-horizon operating targets.

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
  --algorithm goal_conditioned_jepa_ppo \
  --episodes 1000 \
  --seed 7 \
  --output-dir results/agent_training/goal_conditioned_jepa_ppo

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
  --max-steps 610 \
  --mpc-horizon 8 \
  --include-rl-baselines \
  --report-format html,csv,md
```

Docker:
```bash
docker-compose up --build build-agent
```

Run individual stages:
```bash
docker-compose up --build train-agent
docker-compose up --build run-agent
```

Equivalent direct command:
```bash
python -m agents.runners.run_agent \
  --config configs/base_case.json \
  --output-dir results/agent_run \
  --seeds 30 \
  --n-scenarios 6 \
  --model C \
  --max-steps 610 \
  --mpc-horizon 8 \
  --include-rl-baselines \
  --report-format html,csv,md
```

## License and Citation

This repository is currently released for **academic review, transparency, and reproducibility before formal publication**.

The associated paper is:

**Safe Hybrid Agentic Control for Electric Arc Furnace Steelmaking:  
A Multi-Policy Digital Twin Benchmark**

Authors:

- **Vahid Tavakkoli**  
  Institute for Smart System Technologies  
  Universitaet Klagenfurt  
  Klagenfurt, Austria  
  vahid.tavakkoli@aau.at

- **Kabeh Mohsenzadegan**  
  Institute for Smart System Technologies  
  Universitaet Klagenfurt  
  Klagenfurt, Austria  
  kabeh.mohsenzadegan@aau.at

- **Kyandoghere Kyamakya**  
  Universitaet Klagenfurt / Institute for Smart Systems Technologies, Austria  
  and Faculte Polytechnique, Universite de Kinshasa, DR-Congo  
  kyandoghere.kyamakya@aau.at

Repository:

https://github.com/vtavakkoli/eaf-cognitive-twin

### Citation

If you use this repository, benchmark, methodology, results, figures, tables, or code, please cite the associated paper. Before formal publication, cite this repository as:

```bibtex
@misc{tavakkoli2026eafcognitivetwin,
  author       = {Vahid Tavakkoli and Kabeh Mohsenzadegan and Kyandoghere Kyamakya},
  title        = {Safe Hybrid Agentic Control for Electric Arc Furnace Steelmaking: A Multi-Policy Digital Twin Benchmark},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/vtavakkoli/eaf-cognitive-twin}
}
```
