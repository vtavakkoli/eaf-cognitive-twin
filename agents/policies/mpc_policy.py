from __future__ import annotations

import copy

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class MPCPolicy(BasePolicy):
    """Constrained MPC policy using Model C short-horizon rollouts."""

    name = "mpc"

    def __init__(self, horizon: int = 8):
        self.horizon = max(1, horizon)

    def _candidates(self, tap_now: bool) -> list[ActionDict]:
        return [
            {"power_mw": 90.0, "oxygen_nm3_min": 70.0, "ng_nm3_min": 16.0, "carbon_kg_min": 14.0, "flux_kg_min": 110.0, "tap_command": False},
            {"power_mw": 70.0, "oxygen_nm3_min": 55.0, "ng_nm3_min": 12.0, "carbon_kg_min": 10.0, "flux_kg_min": 90.0, "tap_command": False},
            {"power_mw": 48.0, "oxygen_nm3_min": 34.0, "ng_nm3_min": 8.0, "carbon_kg_min": 8.0, "flux_kg_min": 70.0, "tap_command": False},
            {"power_mw": 22.0, "oxygen_nm3_min": 10.0, "ng_nm3_min": 2.0, "carbon_kg_min": 1.0, "flux_kg_min": 15.0, "tap_command": tap_now},
            {"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0, "tap_command": tap_now},
        ]

    def _rollout_score(self, observation: ObservationDict, action: ActionDict) -> float:
        model = observation.get("_model_obj")
        state = observation.get("_state_obj")
        cfg = observation.get("_config_obj")
        if model is None or state is None or cfg is None:
            return -1e9
        sim_state = copy.deepcopy(state)
        score = 0.0
        prev_temp = float(observation["bath_temp_c"])
        for _ in range(self.horizon):
            extras = model._step_dynamics(sim_state, action, [])
            temp_c = sim_state.steel_temp_k - 273.15
            score += sim_state.cum_tapped_kg / max(cfg.tap_target_steel_kg, 1e-9)
            score -= 0.3 * abs(temp_c - cfg.tap_target_temp_c)
            score -= 0.1 * (action["power_mw"] + 0.4 * action["oxygen_nm3_min"] + 0.2 * action["ng_nm3_min"])
            score -= 8.0 * abs(sim_state.steel_carbon_wt_pct - 0.05)
            if temp_c > min(cfg.max_temp_c, 1850.0) + 1e-6:
                return -1e8
            if temp_c > 1700.0 and sim_state.tap_end_time_s is None:
                score -= 40.0
            if temp_c - prev_temp > 50.0:
                score -= 15.0
            prev_temp = temp_c
            if extras.get("tapped_kg_s", 0.0) > 0:
                score += 100.0
                break
        return score

    def act(self, observation: ObservationDict) -> ActionDict:
        can_tap = bool(observation.get("can_tap", False))
        tap_now = can_tap
        candidates = self._candidates(tap_now)
        return max(candidates, key=lambda a: self._rollout_score(observation, a))
