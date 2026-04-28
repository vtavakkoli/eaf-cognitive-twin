from __future__ import annotations

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class IndustrialBaselineSchedulePolicy(BasePolicy):
    name = "baseline_schedule"

    def __init__(self, tap_mass_tolerance: float = 0.95):
        self.tap_mass_tolerance = tap_mass_tolerance

    def _tap_decision(self, observation: ObservationDict) -> tuple[bool, str]:
        cfg = observation.get("_config_obj")
        if cfg is None:
            return False, "not_ready"
        temp = float(observation["bath_temp_c"])
        mass = float(observation["liquid_steel_kg"])
        carbon = float(observation.get("steel_carbon_wt_pct", 0.05))
        t = float(observation.get("time_min", 0.0))

        enough_mass = mass >= cfg.tap_target_steel_kg * self.tap_mass_tolerance
        if not enough_mass:
            return False, "insufficient_liquid_steel"
        in_carbon = 0.02 <= carbon <= 0.08
        if not in_carbon:
            return False, "carbon_out_of_range"

        if temp >= cfg.tap_target_temp_c:
            return True, "target_temp_reached"

        tap_min_temp = getattr(cfg, "tap_min_temp_c", cfg.tap_target_temp_c - 20.0)
        max_heat = getattr(cfg, "max_heat_time_min", cfg.heat_duration_min)
        if t >= max_heat and temp >= tap_min_temp:
            return True, "max_heat_time_reached"
        return False, "not_ready"

    def act(self, observation: ObservationDict) -> ActionDict:
        recipe = observation.get("default_schedule_action")
        if recipe is None:
            recipe = {
                "power_mw": 60.0,
                "oxygen_nm3_min": 45.0,
                "ng_nm3_min": 10.0,
                "carbon_kg_min": 10.0,
                "flux_kg_min": 100.0,
            }
        tap, _ = self._tap_decision(observation)
        return {
            **recipe,
            "tap_command": tap,
        }

    def tap_reason(self, observation: ObservationDict) -> str:
        return self._tap_decision(observation)[1]
