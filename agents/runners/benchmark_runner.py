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
    max_steps: int = 650,
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
                outcome = run_episode(controller, actual_policy, policy_name=policy_name, max_steps=max_steps)
                ts_path = output_dir / "timeseries" / f"agent_timeseries_{scen_name}_{policy_name}_seed{seed}.csv"
                outcome.episode_df.to_csv(ts_path, index=False)
                last = outcome.episode_df.iloc[-1]
                tap_reason = str(last.get("tap_reason", "not_ready"))
                tapped_kg = float(last.get("liquid_steel_kg", 0.0))
                target = float(controller.config.tap_target_steel_kg)
                tapped_tons = tapped_kg / 1000.0
                energy_per_ton = float(last["cum_electric_mwh"]) / tapped_tons
                oxygen_per_ton = float(last["cum_oxygen_nm3"]) / tapped_tons
                ng_per_ton = float(last["cum_ng_nm3"]) / tapped_tons
                temp_err = abs(float(last["bath_temp_c"]) - float(controller.config.tap_target_temp_c))
                mass_err = abs(tapped_kg - target)
                carbon_err = abs(float(last["steel_carbon_wt_pct"]) - 0.05)
                violation_cnt = int(outcome.episode_df.get("safety_violation", pd.Series(dtype=bool)).sum()) + int(outcome.episode_df.get("temperature_violation", pd.Series(dtype=bool)).sum())
                invalid_cnt = int(outcome.episode_df.get("invalid_tap_command", pd.Series(dtype=bool)).sum()) + int(outcome.episode_df.get("action_clamped", pd.Series(dtype=bool)).sum())
                max_temp_c = float(outcome.episode_df["bath_temp_c"].max())
                final_temp_c = float(last["bath_temp_c"])
                tap_target_temp_c = float(controller.config.tap_target_temp_c)
                reached_tap_temp = bool(max_temp_c >= tap_target_temp_c)
                max_melted_fraction = float(outcome.episode_df.get("melted_fraction", pd.Series([0.0])).max())
                can_tap_ever_true = bool(outcome.episode_df.get("can_tap", pd.Series(dtype=bool)).fillna(False).astype(bool).any())
                can_tap_final = bool(last.get("can_tap", False))
                final_liquid_steel_kg = float(last.get("liquid_steel_kg", 0.0))
                tappable_molten_kg = final_liquid_steel_kg
                tap_command_ever_true = bool(outcome.episode_df.get("tap_command", pd.Series(dtype=bool)).fillna(False).astype(bool).any())
                tap_blocked_by_safety_filter_count = int(outcome.episode_df.get("invalid_tap_command", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
                termination_reason = str(last.get("termination_reason", "tapped" if tapped_kg > 0.0 else "heat_end_without_tap"))
                melt_temp_c = float(controller.config.steel_melt_temp_k - 273.15)
                tap_success = bool(tapped_kg > 0.0 or final_temp_c >= melt_temp_c)
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
                        "cum_tapped_kg": tapped_kg,
                        "tappable_molten_kg": tappable_molten_kg,
                        "final_liquid_steel_kg": final_liquid_steel_kg,
                        "tapped_t": tapped_kg / 1000.0,
                        "tap_success": tap_success,
                        "final_temp_c": final_temp_c,
                        "cum_electric_mwh": float(last["cum_electric_mwh"]),
                        "cum_oxygen_nm3": float(last["cum_oxygen_nm3"]),
                        "cum_ng_nm3": float(last["cum_ng_nm3"]),
                        "final_carbon_wt_pct": float(last["steel_carbon_wt_pct"]),
                        "safety_violation_count": int(outcome.episode_df.get("safety_violation", pd.Series(dtype=bool)).sum()),
                        "temperature_violation_count": int(outcome.episode_df.get("temperature_violation", pd.Series(dtype=bool)).sum()),
                        "invalid_tap_count": int(outcome.episode_df.get("invalid_tap_command", pd.Series(dtype=bool)).sum()),
                        "action_clamp_count": int(outcome.episode_df.get("action_clamped", pd.Series(dtype=bool)).sum()),
                        "max_bath_temp_c": max_temp_c,
                        "reached_tap_temp": reached_tap_temp,
                        "final_bath_temp_c": final_temp_c,
                        "max_melted_fraction": max_melted_fraction,
                        "can_tap_ever_true": can_tap_ever_true,
                        "tap_command_ever_true": tap_command_ever_true,
                        "tap_blocked_by_safety_filter_count": tap_blocked_by_safety_filter_count,
                        "termination_reason": termination_reason,
                        "baseline_tap_success_rate": 1.0 if policy_name == "baseline_schedule" and tap_success else 0.0,
                        "baseline_tapped_kg": tapped_kg if policy_name == "baseline_schedule" else 0.0,
                        "baseline_final_temp_c": float(last["bath_temp_c"]) if policy_name == "baseline_schedule" else 0.0,
                        "baseline_tap_reason": tap_reason if policy_name == "baseline_schedule" else "n/a",
                        "energy_per_ton": energy_per_ton,
                        "oxygen_per_ton": oxygen_per_ton,
                        "natural_gas_per_ton": ng_per_ton,
                        "tap_temperature_error": temp_err,
                        "mass_error": mass_err,
                        "carbon_error": carbon_err,
                        "constraint_violation_rate": violation_cnt / max(outcome.steps, 1),
                        "invalid_action_rate": invalid_cnt / max(outcome.steps, 1),
                    }
                )
    return pd.DataFrame(rows)
