# EAF Cognitive Twin Simulator

Research-oriented dynamic simulator for a typical **100 t scrap-based Electric Arc Furnace (EAF)** heat.

## Features
- Three model fidelities in one executable program:
  - **Model A**: empirical reduced-order dynamic model
  - **Model B**: first-principles lumped mass/energy dynamic model
  - **Model C**: enhanced hybrid model (foamy slag, stage-dependent efficiencies, off-gas, optional correction/noise)
- Multi-stage operation: charging, bore-in, main melting, refining, superheating, tapping.
- Time-varying profiles for electric power, oxygen, natural gas, carbon injection, and flux.
- Scenario suite (base + five required variants).
- Sensitivity analysis for key parameters.
- CSV, JSON, and PNG outputs for each run.

## Run locally
```bash
python eaf_simulator.py --output-dir outputs --dt 2.0
```

Optional flags:
- `--config path/to/config.json`
- `--noise 2.0`
- `--seed 123`

## Run via Docker Compose
```bash
docker compose up --build full-run
```

## Expected output files
Under `outputs/`:
- `timeseries_<scenario>_<model>.csv`
- `summary_all_scenarios.csv`
- `summary_all_scenarios.json`
- `resolved_config.json`
- `plot_<scenario>_<model>_*.png`
- `plot_compare_models_<scenario>.png`
- `sensitivity_table.csv`
- `plot_sensitivity_ranking.png`

## Notes
- Numerical integration uses explicit Euler (default `dt=2 s`), with checks for nonphysical states.
- Results are deterministic unless measurement noise is enabled.

## Run unit tests
```bash
python -m unittest discover -s tests -v
```
