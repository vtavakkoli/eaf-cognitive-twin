# Contributing to EAF Cognitive Twin

Thank you for helping improve EAF Cognitive Twin. This repository is a scientific benchmark, so contributions should preserve both software quality and experimental comparability.

## Before you contribute

Please review:

- [`README.md`](README.md) for scope and benchmark commands;
- [`docs/architecture.md`](docs/architecture.md) for component boundaries;
- [`docs/assumptions.md`](docs/assumptions.md) for modelling assumptions;
- [`docs/reproducibility.md`](docs/reproducibility.md) for the experiment protocol;
- [`LICENSE`](LICENSE) for the current research-use terms.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the complete test suite before submitting a change:

```bash
python -m unittest discover -s tests -v
```

A short CLI smoke test is also recommended:

```bash
python -m eaf_twin.cli run \
  --config configs/base_case.json \
  --output-dir outputs/smoke
```

## Scientific change policy

Changes to any of the following require explicit scientific justification in the pull request:

- process equations or physical constants;
- simulator state transitions;
- constraints, safety filters, or tap logic;
- observation/action definitions;
- reward functions or objective weights;
- scenario generation or randomization;
- benchmark metrics or aggregation;
- training/evaluation seed handling;
- default controller parameters;
- reported benchmark results.

For those changes, describe **what changed, why it is justified, and how it affects comparability with previous runs**.

## Reproducibility expectations

When a pull request changes benchmark behaviour:

1. record the configuration used for validation;
2. state the simulator fidelity (A, B, or C);
3. record relevant seeds;
4. preserve raw outputs when practical;
5. compare against the previous behaviour under the same conditions;
6. add or update automated tests;
7. avoid reporting a single favourable run as representative evidence.

## Pull request checklist

A good pull request should:

- have a focused scope and clear title;
- include tests for new or changed behaviour;
- keep public APIs backward compatible when practical;
- update documentation when commands, assumptions, or outputs change;
- avoid committing generated caches, large model binaries, credentials, or plant-confidential data;
- identify any result that is simulation-only rather than plant-validated;
- note whether existing benchmark results remain comparable.

## Coding guidance

- Target Python 3.12+.
- Prefer typed, modular, testable functions over large scripts.
- Keep deterministic behaviour deterministic when a seed is supplied.
- Keep units explicit in names or documentation where ambiguity is possible.
- Validate user/configuration inputs at boundaries.
- Do not weaken safety constraints merely to improve a benchmark score.

## Commit and PR style

Use concise, descriptive commit messages, for example:

```text
fix: preserve seed determinism in scenario generation
feat: add controller ablation configuration
docs: document Model C validation assumptions
test: cover safety-filter boundary conditions
```

## Data and confidentiality

Do not submit proprietary plant data, credentials, personal data, export-controlled material, or information you are not authorized to publish. Synthetic or appropriately licensed public data should be clearly identified as such.

## Reporting problems

For ordinary bugs or reproducibility issues, open a GitHub issue with a minimal reproducer, environment details, configuration, and relevant logs. For security-sensitive reports, follow [`SECURITY.md`](SECURITY.md).
