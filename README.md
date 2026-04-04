# EAF Cognitive Digital Twin

Refactored research-grade foundation for an Electric Arc Furnace (EAF) cognitive digital twin with three fidelity levels:
1. empirical
2. first-principles
3. enhanced hybrid

## Quick start (local)
```bash
pip install -r requirements.txt
pip install -e .
python -m eaf_twin.cli run --config configs/base_case.json --output-dir outputs
```


## Docker
```bash
docker compose up --build full-run
```

## Key upgrades
- Modular package layout under `src/eaf_twin/`.
- Explicit tapped steel tracking (`cum_tapped_kg`, tap start/end).
- Correct useful heat accumulation as stepwise integrated useful heat.
- DRI modeled as charged material with mass and thermal sinks.
- Cleaner mass/energy split (electrical, chemical, useful, losses, off-gas).
- Robust config validation with clear errors.
- Added unit tests and project docs.

## Outputs
- Time series CSV per scenario and model are generated automatically by `docker compose up --build full-run` (or `python -m eaf_twin.cli run --config configs/base_case.json --output-dir outputs`).
- Summary CSV + JSON with upgraded KPIs are generated automatically in `outputs/`.
- PNG trend plots are generated automatically in `outputs/`.
- Generated files under `outputs/` are runtime artifacts and are not meant to be committed.

## Tests
```bash
python -m unittest discover -s tests -v
```
