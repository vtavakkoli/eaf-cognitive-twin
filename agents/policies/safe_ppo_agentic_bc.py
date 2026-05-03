from __future__ import annotations

from pathlib import Path

from agents.base import BasePolicy
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.rl_common import ACTION_NAMES, safe_discrete_action
from agents.types import ActionDict, ObservationDict


class SafePPOAgenticBCPolicy(BasePolicy):
    """Hybrid policy: BC proposal -> PPO residual refinement -> safety supervisor."""

    name = "safe_ppo_agentic_bc"

    def __init__(self, ppo_policy: PPOPolicy | None = None, bc_policy: BehaviorCloningPolicy | None = None):
        self.ppo = ppo_policy or PPOPolicy()
        self.bc = bc_policy or BehaviorCloningPolicy()
        self.last_info: dict[str, object] = {}

    @classmethod
    def load(cls, ppo_path: Path, bc_path: Path) -> "SafePPOAgenticBCPolicy":
        return cls(ppo_policy=PPOPolicy.load(ppo_path), bc_policy=BehaviorCloningPolicy.load(bc_path))

    def _residual_refine(self, observation: ObservationDict, bc_action: ActionDict) -> ActionDict:
        bc_idx = min(
            range(len(ACTION_NAMES)),
            key=lambda i: abs(safe_discrete_action(ACTION_NAMES[i], observation)["power_mw"] - float(bc_action.get("power_mw", 0.0))),
        )
        probs = self.ppo.probs(observation)
        ppo_idx = int(probs.argmax())
        residual_idx = int(round(0.6 * bc_idx + 0.4 * ppo_idx))
        residual_idx = max(0, min(len(ACTION_NAMES) - 1, residual_idx))
        return safe_discrete_action(ACTION_NAMES[residual_idx], observation)

    def _safety_supervisor(self, observation: ObservationDict, action: ActionDict) -> tuple[ActionDict, str]:
        out = dict(action)
        reason = "none"
        can_tap = bool(observation.get("can_tap", False))
        temp_c = float(observation.get("bath_temp_c", 0.0))
        max_temp_c = float(observation.get("_config_obj").max_temp_c if observation.get("_config_obj") else 1700.0)

        if bool(out.get("tap_command", False)) and not can_tap:
            out["tap_command"] = False
            reason = "blocked_invalid_tap"

        if temp_c >= max_temp_c - 8.0:
            out["power_mw"] = min(float(out.get("power_mw", 0.0)), 12.0)
            out["oxygen_nm3_min"] = min(float(out.get("oxygen_nm3_min", 0.0)), 8.0)
            out["ng_nm3_min"] = min(float(out.get("ng_nm3_min", 0.0)), 3.0)
            reason = "overheat_prevention"

        if bool(observation.get("is_downtime", False)):
            out.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0, "tap_command": False})
            reason = "downtime_hold"

        if can_tap and bool(out.get("tap_command", False)):
            out.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0})
        return out, reason

    def act(self, observation: ObservationDict) -> ActionDict:
        bc_action = self.bc.act(observation)
        refined = self._residual_refine(observation, bc_action)
        final_action, reason = self._safety_supervisor(observation, refined)
        self.last_info = {
            "bc_proposal": dict(bc_action),
            "ppo_refined": dict(refined),
            "final_action": dict(final_action),
            "safety_reason": reason,
            "pipeline": "state->bc->ppo_residual->safety_supervisor->final",
        }
        return final_action
