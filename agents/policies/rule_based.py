from __future__ import annotations

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class RuleBasedPolicy(BasePolicy):
    name = "rule_based"

    def act(self, observation: ObservationDict) -> ActionDict:
        melt = float(observation["melted_fraction"])
        temp = float(observation["bath_temp_c"])
        can_tap = bool(observation.get("can_tap", False))

        if melt < 0.4:
            return {"power_mw": 95.0, "oxygen_nm3_min": 75.0, "ng_nm3_min": 24.0, "carbon_kg_min": 20.0, "flux_kg_min": 150.0, "tap_command": False}
        if melt < 0.9:
            return {"power_mw": 78.0, "oxygen_nm3_min": 60.0, "ng_nm3_min": 16.0, "carbon_kg_min": 14.0, "flux_kg_min": 120.0, "tap_command": False}
        tap = can_tap and temp >= 1600.0
        return {"power_mw": 35.0, "oxygen_nm3_min": 24.0, "ng_nm3_min": 5.0, "carbon_kg_min": 4.0, "flux_kg_min": 40.0, "tap_command": tap}
