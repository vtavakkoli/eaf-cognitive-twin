from __future__ import annotations

import copy
from pathlib import Path

from agents.base import BasePolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.types import ActionDict, ObservationDict


class SafePPOAgenticMPCPolicy(BasePolicy):
    """Proposed hybrid: PPO proposal + model-based local correction + safe tap gating."""

    name = "safe_ppo_agentic_mpc"

    def __init__(self, ppo_policy: PPOPolicy | None = None, horizon: int = 3):
        self.ppo = ppo_policy or PPOPolicy()
        self.horizon = max(1, horizon)
        self.last_info: dict[str, object] = {}

    @classmethod
    def load(cls, path: Path, horizon: int = 3) -> "SafePPOAgenticMPCPolicy":
        return cls(ppo_policy=PPOPolicy.load(path), horizon=horizon)

    def _neighbors(self, action: ActionDict) -> list[ActionDict]:
        cands = [dict(action)]
        for scale in (0.85, 1.0, 1.15):
            cands.append(
                {
                    **action,
                    "power_mw": float(action.get("power_mw", 0.0)) * scale,
                    "oxygen_nm3_min": float(action.get("oxygen_nm3_min", 0.0)) * scale,
                    "ng_nm3_min": float(action.get("ng_nm3_min", 0.0)) * scale,
                    "carbon_kg_min": float(action.get("carbon_kg_min", 0.0)) * scale,
                    "flux_kg_min": float(action.get("flux_kg_min", 0.0)) * scale,
                    "tap_command": bool(action.get("tap_command", False)),
                }
            )
        return cands

    def _predict_reward(self, obs: ObservationDict, action: ActionDict) -> float:
        model = obs.get("_model_obj")
        state = obs.get("_state_obj")
        cfg = obs.get("_config_obj")
        if model is None or state is None or cfg is None:
            return -1e9
        s = copy.deepcopy(state)
        total = 0.0
        for _ in range(self.horizon):
            model.apply_charge_events(s, max(0.0, s.time_s - cfg.dt_s), s.time_s)
            model._step_dynamics(s, action, [])
            s.time_s += cfg.dt_s
            temp = s.steel_temp_k - 273.15
            total += 25.0 * s.melted_fraction
            total += 45.0 * min(1.0, s.liquid_steel_kg / max(cfg.tap_target_steel_kg, 1e-9))
            total -= 0.10 * abs(temp - cfg.tap_target_temp_c)
            total -= 0.02 * (action["power_mw"] + 0.2 * action["oxygen_nm3_min"] + 0.2 * action["ng_nm3_min"])
            if temp > cfg.max_temp_c:
                total -= 400.0
        return float(total)

    def act(self, observation: ObservationDict) -> ActionDict:
        ppo_action = self.ppo.act(observation)
        can_tap = bool(observation.get("can_tap", False))
        time_min = float(observation.get("time_min", 0.0))

        strategy = "ppo"
        reason = "none"
        selected = dict(ppo_action)

        if bool(selected.get("tap_command", False)) and not can_tap:
            selected["tap_command"] = False
            strategy = "safe_tap"
            reason = "tap_blocked_not_ready"

        if 66.0 <= time_min <= 70.0 and can_tap:
            selected["tap_command"] = True
            strategy = "safe_tap"
            reason = "tap_window_ready"

        best_reward = self._predict_reward(observation, selected)
        best = dict(selected)
        for cand in self._neighbors(selected):
            if bool(cand.get("tap_command", False)) and not can_tap:
                cand["tap_command"] = False
            r = self._predict_reward(observation, cand)
            if r > best_reward:
                best_reward = r
                best = cand
                strategy = "mpc_correction"
                reason = "local_model_rollout"

        if bool(observation.get("is_downtime", False)):
            best.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0, "tap_command": False})
            strategy = "hold"
            reason = "downtime"

        self.last_info = {
            "ppo_raw_action": dict(ppo_action),
            "mpc_corrected_action": dict(best),
            "correction_reason": reason,
            "predicted_candidate_reward": best_reward,
            "selected_strategy": strategy,
        }
        return best
