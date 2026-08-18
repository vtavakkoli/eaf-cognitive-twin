# Reproducibility Protocol

This document defines the minimum information that should accompany experiments produced with EAF Cognitive Twin. The goal is to make controller comparisons repeatable, auditable, and scientifically interpretable.

## 1. Freeze the software state

Record the exact Git commit used for every reported experiment:

```bash
git rev-parse HEAD
```

If local changes exist, either commit them on a dedicated branch or archive the diff together with the experiment artifacts.

## 2. Freeze the environment

The reference runtime is Python 3.12+ with dependencies pinned in `requirements.txt` and `pyproject.toml`.

Recommended options:

```bash
python --version
python -m pip freeze > environment-freeze.txt
```

or use the provided Docker configuration. Record the host/accelerator information when training performance or wall-clock measurements are reported.

## 3. Archive the exact configuration

Store the complete configuration used for the run. Do not rely only on a verbal description of changed parameters.

At minimum, preserve:

- selected digital-twin model (A, B, or C);
- simulation time step and episode horizon;
- measurement-noise setting;
- process/scenario parameters;
- safety and termination constraints;
- controller-specific hyperparameters;
- training episodes and optimization settings;
- report format and metric definitions.

## 4. Separate training and evaluation randomness

Training seeds and evaluation seeds serve different purposes and should be recorded separately.

For comparative evaluation:

- use the same evaluation scenarios and seeds for all compared policies;
- avoid selecting only favourable seeds after observing results;
- report the number of scenarios and repeated seeds;
- keep stochastic sensor/process settings identical across methods unless uncertainty is the variable under study.

Example evaluation pattern:

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

## 5. Preserve benchmark fairness

A fair comparison changes the policy while holding the evaluation environment fixed. In particular, keep the following constant unless they are explicitly part of an ablation:

- simulator fidelity;
- scenario definitions;
- evaluation seeds;
- observation availability;
- action bounds;
- safety-filter rules;
- termination logic;
- metric definitions;
- episode budget.

If a controller requires additional information or compute that another controller does not receive, disclose that difference.

## 6. Report distributions, not isolated runs

Single-run results can be misleading for stochastic optimization and simulation. Prefer repeated evaluations and report appropriate aggregate statistics together with the number of runs.

Keep raw per-run data available so aggregate tables and figures can be regenerated.

## 7. Distinguish model fidelity from real-world validation

Models A, B, and C represent increasing fidelity **within this repository**. Model C being the highest-fidelity simulator does not imply certified correspondence to a specific production furnace.

When publishing results, clearly distinguish among:

- algorithm performance in simulation;
- calibration against external or plant data;
- retrospective plant-data validation;
- shadow-mode testing;
- closed-loop deployment.

Do not describe simulation-only evidence as plant validation.

## 8. Minimum experiment manifest

For publication-quality runs, store a small manifest next to the output directory. A human-readable example is:

```yaml
repository: vtavakkoli/eaf-cognitive-twin
commit: <git-sha>
python: "3.12.x"
model: C
config: configs/base_case.json
algorithm: ppo
training_seed: 7
evaluation_seeds: 30
scenarios: 6
max_steps: 610
notes: "Main benchmark configuration"
```

Do not place secrets or confidential plant identifiers in the manifest.

## 9. Suggested artifact bundle

A complete experiment archive should contain, when applicable:

```text
experiment/
├── manifest.yaml
├── config.json
├── environment-freeze.txt
├── raw/
├── metrics/
├── figures/
└── report/
```

Generated artifacts should be treated as outputs of a particular software/configuration state, not as timeless ground truth.

## 10. Reproducing CI-level validation

Run:

```bash
python -m pip check
python -m compileall -q src agents
python -m unittest discover -s tests -v
```

Then perform a short simulation smoke test. The GitHub Actions workflow mirrors these checks for pull requests and changes to `main`.

## Citation

When results from this benchmark are used in academic work, cite the preferred paper listed in the repository `README.md` and `CITATION.cff`, and cite the software repository/version when appropriate for computational reproducibility.
