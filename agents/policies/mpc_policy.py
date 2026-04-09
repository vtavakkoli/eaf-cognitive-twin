from __future__ import annotations

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class MPCPolicy(BasePolicy):
    """Lightweight MPC-style controller using a one-step surrogate objective."""

    name = "mpc"

    def _score_action(self, obs: ObservationDict, action: ActionDict) -> float:
        melt = float(obs["melted_fraction"])
        temp = float(obs["bath_temp_c"])
        time_min = float(obs.get("time_min", 0.0))
        remaining_solid = float(obs.get("remaining_solid_kg", 0.0))

        # Surrogate "next-step" tendencies.
        predicted_melt = melt + 0.0018 * float(action["power_mw"]) + 0.0004 * float(action["oxygen_nm3_min"])
        predicted_temp = temp + 0.05 * float(action["power_mw"]) + 0.015 * float(action["oxygen_nm3_min"]) - 0.02 * float(action["flux_kg_min"])
        predicted_remaining_solid = max(0.0, remaining_solid - (0.8 * float(action["power_mw"]) + 0.3 * float(action["oxygen_nm3_min"])))

        progress_reward = 4.0 * min(predicted_melt, 1.0)
        temp_penalty = abs(predicted_temp - 1645.0) / 35.0
        energy_penalty = 0.006 * float(action["power_mw"]) + 0.002 * float(action["oxygen_nm3_min"]) + 0.001 * float(action["ng_nm3_min"])
        late_tap_penalty = 0.0
        if time_min > 50.0 and not bool(action["tap_command"]):
            late_tap_penalty = 1.5
        solid_penalty = predicted_remaining_solid / 35_000.0

        return progress_reward - temp_penalty - energy_penalty - late_tap_penalty - solid_penalty

    def act(self, observation: ObservationDict) -> ActionDict:
        melt = float(observation["melted_fraction"])
        temp = float(observation["bath_temp_c"])
        time_min = float(observation.get("time_min", 0.0))
        can_tap = bool(observation.get("can_tap", False))
        tap_now = can_tap or (melt > 0.97 and temp > 1600.0 and time_min > 48.0)

        candidates: list[ActionDict] = [
            {"power_mw": 108.0, "oxygen_nm3_min": 88.0, "ng_nm3_min": 24.0, "carbon_kg_min": 20.0, "flux_kg_min": 150.0, "tap_command": False},
            {"power_mw": 92.0, "oxygen_nm3_min": 72.0, "ng_nm3_min": 18.0, "carbon_kg_min": 16.0, "flux_kg_min": 125.0, "tap_command": False},
            {"power_mw": 70.0, "oxygen_nm3_min": 50.0, "ng_nm3_min": 12.0, "carbon_kg_min": 11.0, "flux_kg_min": 90.0, "tap_command": False},
            {"power_mw": 45.0, "oxygen_nm3_min": 30.0, "ng_nm3_min": 7.0, "carbon_kg_min": 6.0, "flux_kg_min": 60.0, "tap_command": tap_now},
            {"power_mw": 28.0, "oxygen_nm3_min": 14.0, "ng_nm3_min": 3.0, "carbon_kg_min": 2.0, "flux_kg_min": 20.0, "tap_command": tap_now},
        ]
        if time_min < 15.0:
            return candidates[0]
        if melt < 0.85:
            return max(candidates[:3], key=lambda a: self._score_action(observation, a))
        return max(candidates, key=lambda a: self._score_action(observation, a))
