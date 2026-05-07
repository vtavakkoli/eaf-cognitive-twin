# Migration Note

- Monolithic `eaf_simulator.py` logic has been migrated into `src/eaf_twin/`.
- Legacy script remains as a compatibility wrapper.
- Primary command is now `python -m eaf_twin.cli run ...`.
- Configs are now validated and a baseline is provided at `configs/base_case.json`.
