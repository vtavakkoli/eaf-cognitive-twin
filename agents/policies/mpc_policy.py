from __future__ import annotations

import copy

from eaf_twin.simulation.schedule import active_setpoints

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class MPCPolicy(BasePolicy):
    """Short-horizon controller anchored to the configured industrial schedule."""

    name = "mpc"

    def __init__(self, horizon: int = 8):
        self.horizon = max(1, horizon)

    @staticmethod
    def _clip(action: ActionDict) -> ActionDict:
        return {
            "power_mw": max(0.0, float(action.get("power_mw", 0.0))),
            "oxygen_nm3_min": max(0.0, float(action.get("oxygen_nm3_min", 0.0))),
            "ng_nm3_min": max(0.0, float(action.get("ng_nm3_min", 0.0))),
            "carbon_kg_min": max(0.0, float(action.get("carbon_kg_min", 0.0))),
            "flux_kg_min": max(0.0, float(action.get("flux_kg_min", 0.0))),
            "tap_command": bool(action.get("tap_command", False)),
        }

    def _default_action(self, observation: ObservationDict, time_min: float | None = None) -> ActionDict:
        cfg = observation.get("_config_obj")
        if cfg is not None and time_min is not None:
            base = active_setpoints(cfg, time_min) | {"tap_command": False}
            return self._clip(base)
        default_action = dict(observation.get("default_schedule_action", {}))
        if not default_action:
            default_action = {"power_mw": 78.0, "oxygen_nm3_min": 70.0, "ng_nm3_min": 16.0, "carbon_kg_min": 20.0, "flux_kg_min": 100.0}
        default_action["tap_command"] = False
        return self._clip(default_action)

    def _candidates(self, observation: ObservationDict) -> list[ActionDict]:
        temp_c = float(observation.get("bath_temp_c", 0.0))
        melt = float(observation.get("melted_fraction", 0.0))
        can_tap = bool(observation.get("can_tap", False))
        base = self._default_action(observation)
        cfg = observation.get("_config_obj")
        tap_target = float(getattr(cfg, "tap_target_temp_c", 1645.0))
        temp_error = tap_target - temp_c

        power_bias = 0.18 * temp_error
        if melt < 0.45:
            power_bias += 8.0
        elif melt > 0.9:
            power_bias -= 6.0

        centered = {
            **base,
            "power_mw": base["power_mw"] + power_bias,
            "oxygen_nm3_min": base["oxygen_nm3_min"] + 0.22 * power_bias,
            "ng_nm3_min": base["ng_nm3_min"] + 0.10 * power_bias,
            "carbon_kg_min": base["carbon_kg_min"] + 0.08 * power_bias,
            "flux_kg_min": base["flux_kg_min"],
            "tap_command": False,
        }

        deltas = [-12.0, 0.0, 12.0]
        candidates = []
        for d in deltas:
            act = {
                **centered,
                "power_mw": centered["power_mw"] + d,
                "oxygen_nm3_min": centered["oxygen_nm3_min"] + 0.55 * d,
                "ng_nm3_min": centered["ng_nm3_min"] + 0.18 * d,
                "carbon_kg_min": centered["carbon_kg_min"] + 0.15 * d,
                "flux_kg_min": centered["flux_kg_min"] + 0.60 * d,
                "tap_command": False,
            }
            candidates.append(self._clip(act))

        candidates.append(self._clip(base))
        if can_tap:
            candidates.append(self._clip({**base, "power_mw": max(base["power_mw"] * 0.3, 8.0), "oxygen_nm3_min": max(base["oxygen_nm3_min"] * 0.2, 2.0), "tap_command": True}))
        return candidates

    def _rollout_score(self, observation: ObservationDict, first_action: ActionDict) -> float:
        model = observation.get("_model_obj")
        state = observation.get("_state_obj")
        cfg = observation.get("_config_obj")
        if model is None or state is None or cfg is None:
            return -1e9

        sim_state = copy.deepcopy(state)
        score = 0.0

        for step_idx in range(self.horizon):
            time_min = sim_state.time_s / 60.0
            action = first_action if step_idx == 0 else self._default_action(observation, time_min=time_min)
            t_prev = sim_state.time_s
            model.apply_charge_events(sim_state, max(0.0, t_prev - cfg.dt_s), t_prev)
            extras = model._step_dynamics(sim_state, action, [])
            sim_state.time_s += cfg.dt_s

            temp_c = sim_state.steel_temp_k - 273.15
            temp_error = abs(temp_c - cfg.tap_target_temp_c)
            melt_frac = sim_state.melted_fraction
            liquid_frac = min(sim_state.liquid_steel_kg / max(cfg.tap_target_steel_kg, 1e-9), 1.2)
            tap_frac = min(sim_state.cum_tapped_kg / max(cfg.tap_target_steel_kg, 1e-9), 1.2)
            carbon_error = abs(sim_state.steel_carbon_wt_pct - 0.05)

            score += 40.0 * melt_frac + 45.0 * liquid_frac + 90.0 * tap_frac
            score -= 0.22 * temp_error
            score -= 7.5 * carbon_error
            score -= 0.02 * (action["power_mw"] + 0.3 * action["oxygen_nm3_min"] + 0.2 * action["ng_nm3_min"])

            if temp_c > cfg.max_temp_c:
                return -1e8
            if melt_frac > 0.9 and temp_c < cfg.tap_target_temp_c - 40.0:
                score -= 30.0
            if extras.get("tapped_kg_s", 0.0) > 0.0:
                score += 140.0
                break

        return score

    def act(self, observation: ObservationDict) -> ActionDict:
        time_min = float(observation.get("time_min", 0.0))
        can_tap = bool(observation.get("can_tap", False))
        base = self._default_action(observation)

        if 58.0 <= time_min <= 61.0 and can_tap:
            return self._clip(
                {
                    **base,
                    "power_mw": max(8.0, base["power_mw"] * 0.25),
                    "oxygen_nm3_min": max(2.0, base["oxygen_nm3_min"] * 0.20),
                    "tap_command": True,
                }
            )

        candidates = self._candidates(observation)
        best = max(candidates, key=lambda a: self._rollout_score(observation, a))
        melt = float(observation.get("melted_fraction", 0.0))
        if time_min < 61.0 and melt < 0.98:
            best = self._clip(
                {
                    **best,
                    "power_mw": max(best["power_mw"], base["power_mw"] + 10.0),
                    "oxygen_nm3_min": max(best["oxygen_nm3_min"], base["oxygen_nm3_min"] + 8.0),
                    "ng_nm3_min": max(best["ng_nm3_min"], base["ng_nm3_min"] + 2.0),
                    "carbon_kg_min": max(best["carbon_kg_min"], base["carbon_kg_min"] + 2.0),
                    "tap_command": False,
                }
            )
        return best
