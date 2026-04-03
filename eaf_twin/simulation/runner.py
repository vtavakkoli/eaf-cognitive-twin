from __future__ import annotations

from pathlib import Path

import pandas as pd

from eaf_twin.config.defaults import scenario_configs
from eaf_twin.config.loader import save_config
from eaf_twin.io.persistence import save_summary_table, save_time_series
from eaf_twin.models.empirical import EmpiricalModel
from eaf_twin.models.first_principles import FirstPrinciplesModel
from eaf_twin.reporting.plots import plot_core
from eaf_twin.validation.checks import plausibility_checks
from eaf_twin.validation.metrics import model_status


def run_full_simulation(config, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "resolved_config.json", config)
    rows = []
    for scen_name, scen_cfg in scenario_configs(config).items():
        scen_cfg.heat_name = scen_name
        for model in (EmpiricalModel(scen_cfg), FirstPrinciplesModel(scen_cfg, False), FirstPrinciplesModel(scen_cfg, True)):
            result = model.simulate()
            issues = plausibility_checks(result.summary)
            save_time_series(result.df, output_dir, f"timeseries_{scen_name}_{result.model_name}.csv")
            plot_core(result.df, output_dir, f"{scen_name}_{result.model_name}")
            rows.append({
                "scenario": scen_name,
                "model": result.model_name,
                **result.summary,
                "model_status": model_status(issues),
                "issues": " | ".join(issues),
            })
    save_summary_table(rows, output_dir, "all_scenarios")
    return pd.DataFrame(rows)
