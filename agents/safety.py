from __future__ import annotations

from dataclasses import dataclass

from agents.types import ActionDict


@dataclass
class SafetyLimits:
    power_mw: tuple[float, float] = (0.0, 180.0)
    oxygen_nm3_min: tuple[float, float] = (0.0, 140.0)
    ng_nm3_min: tuple[float, float] = (0.0, 60.0)
    carbon_kg_min: tuple[float, float] = (0.0, 50.0)
    flux_kg_min: tuple[float, float] = (0.0, 250.0)
    ramps_per_min: dict[str, float] = None

    def __post_init__(self) -> None:
        if self.ramps_per_min is None:
            self.ramps_per_min = {
                "power_mw": 15.0,
                "oxygen_nm3_min": 20.0,
                "ng_nm3_min": 12.0,
                "carbon_kg_min": 10.0,
                "flux_kg_min": 35.0,
            }


class SafetyFilter:
    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def apply(self, action: ActionDict, prev_action: ActionDict | None, dt_min: float, can_tap: bool) -> ActionDict:
        safe = dict(action)
        for key in ("power_mw", "oxygen_nm3_min", "ng_nm3_min", "carbon_kg_min", "flux_kg_min"):
            lo, hi = getattr(self.limits, key)
            safe[key] = self._clamp(float(safe.get(key, 0.0)), lo, hi)
            safe[key] = max(0.0, safe[key])
            if prev_action is not None:
                max_delta = self.limits.ramps_per_min[key] * max(dt_min, 1e-6)
                safe[key] = self._clamp(safe[key], float(prev_action[key]) - max_delta, float(prev_action[key]) + max_delta)
        safe["tap_command"] = bool(safe.get("tap_command", False) and can_tap)

        if safe["power_mw"] < 5.0:
            safe["oxygen_nm3_min"] = min(safe["oxygen_nm3_min"], 10.0)
        return safe
