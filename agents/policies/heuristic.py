from __future__ import annotations

from dataclasses import asdict, dataclass

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


@dataclass
class HeuristicParams:
    early_power_mw: float = 92.0
    mid_power_mw: float = 72.0
    late_power_mw: float = 30.0
    oxygen_scale: float = 1.0
    gas_scale: float = 1.0


class HeuristicPolicy(BasePolicy):
    name = "heuristic"

    def __init__(self, params: HeuristicParams | None = None):
        self.params = params or HeuristicParams()

    def to_dict(self) -> dict:
        return asdict(self.params)

    def act(self, observation: ObservationDict) -> ActionDict:
        p = self.params
        melt = float(observation["melted_fraction"])
        temp = float(observation["bath_temp_c"])
        can_tap = bool(observation.get("can_tap", False))
        if melt < 0.5:
            power = p.early_power_mw
        elif melt < 0.92:
            power = p.mid_power_mw
        else:
            power = p.late_power_mw
        oxygen = (68.0 if melt < 0.8 else 28.0) * p.oxygen_scale
        ng = (18.0 if melt < 0.85 else 4.0) * p.gas_scale
        carbon = 16.0 if melt < 0.85 else 5.0
        flux = 130.0 if melt < 0.7 else 45.0
        return {
            "power_mw": power,
            "oxygen_nm3_min": oxygen,
            "ng_nm3_min": ng,
            "carbon_kg_min": carbon,
            "flux_kg_min": flux,
            "tap_command": can_tap and temp >= 1600.0,
        }
