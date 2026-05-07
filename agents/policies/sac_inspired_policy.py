from __future__ import annotations

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class SACInspiredPolicy(BasePolicy):
    name = "sac_inspired"

    def act(self, observation: ObservationDict) -> ActionDict:
        melt = float(observation.get("melted_fraction", 0.0))
        temp = float(observation.get("bath_temp_c", 0.0))
        target = 1645.0
        err = target - temp
        power = 65.0 + 35.0 * (1.0 - melt) + 0.08 * err
        oxygen = 40.0 + 45.0 * (1.0 - melt) + 0.15 * err
        return {
            "power_mw": max(5.0, power),
            "oxygen_nm3_min": max(2.0, oxygen),
            "ng_nm3_min": max(0.0, 4.0 + 20.0 * (1.0 - melt)),
            "carbon_kg_min": max(0.0, 2.0 + 10.0 * (1.0 - melt)),
            "flux_kg_min": max(0.0, 20.0 + 120.0 * (1.0 - melt)),
            "tap_command": bool(observation.get("can_tap", False) and temp > 1600.0),
        }
