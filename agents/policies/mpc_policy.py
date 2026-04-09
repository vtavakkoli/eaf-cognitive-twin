from __future__ import annotations

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class MPCPolicy(BasePolicy):
    """Lightweight MPC-style controller using a one-step surrogate objective."""

    name = "mpc"

    def _score_action(self, obs: ObservationDict, action: ActionDict) -> float:
        melt = float(obs["melted_fraction"])
        temp = float(obs["bath_temp_c"])

        # Surrogate "next-step" tendencies.
        predicted_melt = melt + 0.0018 * float(action["power_mw"]) + 0.0004 * float(action["oxygen_nm3_min"])
        predicted_temp = temp + 0.05 * float(action["power_mw"]) + 0.015 * float(action["oxygen_nm3_min"]) - 0.02 * float(action["flux_kg_min"])

        progress_reward = 4.0 * min(predicted_melt, 1.0)
        temp_penalty = abs(predicted_temp - 1645.0) / 35.0
        energy_penalty = 0.006 * float(action["power_mw"]) + 0.002 * float(action["oxygen_nm3_min"]) + 0.001 * float(action["ng_nm3_min"])

        return progress_reward - temp_penalty - energy_penalty

    def act(self, observation: ObservationDict) -> ActionDict:
        can_tap = bool(observation.get("can_tap", False))
        candidates: list[ActionDict] = [
            {"power_mw": 100.0, "oxygen_nm3_min": 80.0, "ng_nm3_min": 20.0, "carbon_kg_min": 18.0, "flux_kg_min": 140.0, "tap_command": False},
            {"power_mw": 80.0, "oxygen_nm3_min": 58.0, "ng_nm3_min": 14.0, "carbon_kg_min": 14.0, "flux_kg_min": 100.0, "tap_command": False},
            {"power_mw": 45.0, "oxygen_nm3_min": 30.0, "ng_nm3_min": 7.0, "carbon_kg_min": 6.0, "flux_kg_min": 60.0, "tap_command": can_tap},
            {"power_mw": 20.0, "oxygen_nm3_min": 10.0, "ng_nm3_min": 2.0, "carbon_kg_min": 2.0, "flux_kg_min": 20.0, "tap_command": can_tap},
        ]
        return max(candidates, key=lambda a: self._score_action(observation, a))
