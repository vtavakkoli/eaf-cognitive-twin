from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from eaf_twin.config.defaults import scenario_configs
from eaf_twin.config.loader import load_config

from agents.base import BasePolicy
from agents.controller import EAFController
from agents.policies.baseline_schedule import IndustrialBaselineSchedulePolicy
from agents.runners.episode_runner import run_episode


def run_benchmark(
    config_path: Path | None,
    policies: dict[str, BasePolicy],
    output_dir: Path,
    seeds: list[int],
    selected_scenarios: list[str] | None = None,
) -> pd.DataFrame:
    cfg = load_config(config_path) if config_path else load_config(None)
    scenarios = scenario_configs(cfg)
    scenario_names = selected_scenarios or list(scenarios.keys())
    rows = []
    (output_dir / "timeseries").mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        for scen_name in scenario_names:
            scen_cfg = replace(scenarios[scen_name], random_seed=seed)
            for policy_name, policy in policies.items():
                controller = EAFController(replace(scen_cfg), enhanced_model=True)
                actual_policy: BasePolicy = IndustrialBaselineSchedulePolicy() if policy_name == "baseline_schedule" else policy
                outcome = run_episode(controller, actual_policy, policy_name=policy_name)
                ts_path = output_dir / "timeseries" / f"agent_timeseries_{scen_name}_{policy_name}_seed{seed}.csv"
                outcome.episode_df.to_csv(ts_path, index=False)
                last = outcome.episode_df.iloc[-1]
                tap_reason = str(last.get("tap_reason", "not_ready"))
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scen_name,
                        "policy": policy_name,
                        "model_name": controller.model_name,
                        "total_reward": outcome.total_reward,
                        "step_reward_sum": float(outcome.episode_df["reward"].sum()),
                        "terminal_reward": float(last.get("terminal_reward", 0.0)),
                        "steps": outcome.steps,
                        "cum_tapped_kg": float(last["cum_tapped_kg"]),
                        "tapped_t": float(last["cum_tapped_kg"]) / 1000.0,
                        "tap_success": bool(float(last["cum_tapped_kg"]) > 0.0),
                        "final_temp_c": float(last["bath_temp_c"]),
                        "cum_electric_mwh": float(last["cum_electric_mwh"]),
                        "cum_oxygen_nm3": float(last["cum_oxygen_nm3"]),
                        "cum_ng_nm3": float(last["cum_ng_nm3"]),
                        "final_carbon_wt_pct": float(last["steel_carbon_wt_pct"]),
                        "safety_violation_count": int(outcome.episode_df.get("safety_violation", pd.Series(dtype=bool)).sum()),
                        "temperature_violation_count": int(outcome.episode_df.get("temperature_violation", pd.Series(dtype=bool)).sum()),
                        "invalid_tap_count": int(outcome.episode_df.get("invalid_tap_command", pd.Series(dtype=bool)).sum()),
                        "action_clamp_count": int(outcome.episode_df.get("action_clamped", pd.Series(dtype=bool)).sum()),
                        "max_bath_temp_c": float(outcome.episode_df["bath_temp_c"].max()),
                        "baseline_tap_success_rate": 1.0 if policy_name == "baseline_schedule" and float(last["cum_tapped_kg"]) > 0.0 else 0.0,
                        "baseline_tapped_kg": float(last["cum_tapped_kg"]) if policy_name == "baseline_schedule" else 0.0,
                        "baseline_final_temp_c": float(last["bath_temp_c"]) if policy_name == "baseline_schedule" else 0.0,
                        "baseline_tap_reason": tap_reason if policy_name == "baseline_schedule" else "n/a",
                    }
                )
    return pd.DataFrame(rows)
