from __future__ import annotations

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class TD3InspiredPolicy(BasePolicy):
    name = "td3_inspired"

    def act(self, observation: ObservationDict) -> ActionDict:
        melt = float(observation.get("melted_fraction", 0.0))
        temp = float(observation.get("bath_temp_c", 0.0))
        power = 72.0 + 24.0 * (1.0 - melt)
        oxygen = 30.0 + 55.0 * (1.0 - melt)
        if temp > 1680.0:
            power *= 0.55
            oxygen *= 0.45
        return {
            "power_mw": power,
            "oxygen_nm3_min": oxygen,
            "ng_nm3_min": 3.0 + 14.0 * (1.0 - melt),
            "carbon_kg_min": 1.0 + 9.0 * (1.0 - melt),
            "flux_kg_min": 12.0 + 95.0 * (1.0 - melt),
            "tap_command": bool(observation.get("can_tap", False) and temp >= 1600.0 and melt > 0.96),
        }
