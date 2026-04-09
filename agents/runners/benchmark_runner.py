from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from eaf_twin.config.defaults import scenario_configs
from eaf_twin.config.loader import load_config

from agents.base import BasePolicy
from agents.controller import EAFController
from agents.runners.episode_runner import run_episode


def run_benchmark(config_path: Path | None, policies: dict[str, BasePolicy], output_dir: Path) -> pd.DataFrame:
    cfg = load_config(config_path) if config_path else load_config(None)
    scenarios = scenario_configs(cfg)
    rows = []
    (output_dir / "timeseries").mkdir(parents=True, exist_ok=True)
    for scen_name, scen_cfg in scenarios.items():
        for policy_name, policy in policies.items():
            controller = EAFController(replace(scen_cfg), enhanced_model=True)
            if policy_name == "baseline_schedule":
                class _BaselinePolicy(BasePolicy):
                    name = "baseline_schedule"
                    def act(self, observation):
                        return controller.default_schedule_action()
                actual_policy: BasePolicy = _BaselinePolicy()
            else:
                actual_policy = policy
            outcome = run_episode(controller, actual_policy, policy_name=policy_name)
            ts_path = output_dir / "timeseries" / f"agent_timeseries_{scen_name}_{policy_name}.csv"
            outcome.episode_df.to_csv(ts_path, index=False)
            last = outcome.episode_df.iloc[-1]
            rows.append({
                "scenario": scen_name,
                "policy": policy_name,
                "total_reward": outcome.total_reward,
                "steps": outcome.steps,
                "cum_tapped_kg": float(last["cum_tapped_kg"]),
                "final_temp_c": float(last["bath_temp_c"]),
                "cum_electric_mwh": float(last["cum_electric_mwh"]),
                "cum_oxygen_nm3": float(last["cum_oxygen_nm3"]),
                "cum_ng_nm3": float(last["cum_ng_nm3"]),
                "final_carbon_wt_pct": float(last["steel_carbon_wt_pct"]),
            })
    return pd.DataFrame(rows)
