from __future__ import annotations

from dataclasses import dataclass

from agents.types import ActionDict, ObservationDict


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

    def apply(
        self,
        action: ActionDict,
        prev_action: ActionDict | None,
        dt_min: float,
        can_tap: bool,
        observation: ObservationDict,
        max_temp_c: float,
        is_downtime: bool,
    ) -> tuple[ActionDict, dict[str, bool | str]]:
        safe = dict(action)
        flags: dict[str, bool | str] = {
            "safety_violation": False,
            "temperature_violation": False,
            "carbon_violation": False,
            "invalid_tap_command": False,
            "action_clamped": False,
            "clamp_reason": "none",
        }

        for key in ("power_mw", "oxygen_nm3_min", "ng_nm3_min", "carbon_kg_min", "flux_kg_min"):
            original = float(safe.get(key, 0.0))
            lo, hi = getattr(self.limits, key)
            safe[key] = self._clamp(original, lo, hi)
            safe[key] = max(0.0, safe[key])
            if prev_action is not None:
                max_delta = self.limits.ramps_per_min[key] * max(dt_min, 1e-6)
                safe[key] = self._clamp(safe[key], float(prev_action[key]) - max_delta, float(prev_action[key]) + max_delta)
            if abs(safe[key] - original) > 1e-9:
                flags["action_clamped"] = True

        temp_c = float(observation.get("bath_temp_c", 0.0))

        if is_downtime:
            safe.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0})
            flags["action_clamped"] = True
            flags["clamp_reason"] = "downtime"

        temp_c = float(observation.get("bath_temp_c", 0.0))
        carbon = float(observation.get("steel_carbon_wt_pct", 0.0))

        if temp_c > 1700.0:
            safe["power_mw"] = min(safe["power_mw"], 35.0)
            safe["oxygen_nm3_min"] = min(safe["oxygen_nm3_min"], 20.0)
            flags["action_clamped"] = True
            flags["clamp_reason"] = "high_temp_guard"
        if temp_c > 1800.0:
            safe["power_mw"] = min(safe["power_mw"], 15.0)
            safe["oxygen_nm3_min"] = min(safe["oxygen_nm3_min"], 5.0)
            safe["ng_nm3_min"] = min(safe["ng_nm3_min"], 2.0)
            flags["safety_violation"] = True
            flags["action_clamped"] = True
            flags["clamp_reason"] = "overheat_holding"
        if temp_c > max_temp_c:
            safe.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0})
            flags["safety_violation"] = True
            flags["temperature_violation"] = True
            flags["action_clamped"] = True
            flags["clamp_reason"] = "max_temp_exceeded"

        if carbon < 0.01 or carbon > 0.20:
            flags["carbon_violation"] = True

        requested_tap = bool(action.get("tap_command", False))
        safe["tap_command"] = bool(safe.get("tap_command", False) and can_tap)
        if requested_tap and not safe["tap_command"]:
            flags["invalid_tap_command"] = True

        if safe["power_mw"] < 5.0:
            safe["oxygen_nm3_min"] = min(safe["oxygen_nm3_min"], 10.0)
        return safe, flags
