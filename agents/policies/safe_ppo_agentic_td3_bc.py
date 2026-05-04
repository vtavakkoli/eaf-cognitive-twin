from __future__ import annotations

from pathlib import Path

from agents.base import BasePolicy
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.td3_inspired_policy import TD3InspiredPolicy
from agents.types import ActionDict, ObservationDict


class SafePPOAgenticTD3BCPolicy(BasePolicy):
    """Hybrid policy: BC proposal + PPO/TD3 agentic fusion + safety gating."""

    name = "safe_ppo_agentic_td3_bc"

    def __init__(self, ppo_policy: PPOPolicy | None = None, bc_policy: BehaviorCloningPolicy | None = None):
        self.ppo = ppo_policy or PPOPolicy()
        self.bc = bc_policy or BehaviorCloningPolicy()
        self.td3 = TD3InspiredPolicy()
        self.last_info: dict[str, object] = {}

    @classmethod
    def load(cls, ppo_path: Path, bc_path: Path) -> "SafePPOAgenticTD3BCPolicy":
        return cls(ppo_policy=PPOPolicy.load(ppo_path), bc_policy=BehaviorCloningPolicy.load(bc_path))

    def _blend(self, bc_action: ActionDict, ppo_action: ActionDict, td3_action: ActionDict) -> ActionDict:
        # Weighted fusion emphasizing PPO stability, TD3 smoothing, and BC imitation prior.
        out = {
            "power_mw": 0.25 * float(bc_action.get("power_mw", 0.0)) + 0.45 * float(ppo_action.get("power_mw", 0.0)) + 0.30 * float(td3_action.get("power_mw", 0.0)),
            "oxygen_nm3_min": 0.25 * float(bc_action.get("oxygen_nm3_min", 0.0)) + 0.45 * float(ppo_action.get("oxygen_nm3_min", 0.0)) + 0.30 * float(td3_action.get("oxygen_nm3_min", 0.0)),
            "ng_nm3_min": 0.25 * float(bc_action.get("ng_nm3_min", 0.0)) + 0.45 * float(ppo_action.get("ng_nm3_min", 0.0)) + 0.30 * float(td3_action.get("ng_nm3_min", 0.0)),
            "carbon_kg_min": 0.25 * float(bc_action.get("carbon_kg_min", 0.0)) + 0.45 * float(ppo_action.get("carbon_kg_min", 0.0)) + 0.30 * float(td3_action.get("carbon_kg_min", 0.0)),
            "flux_kg_min": 0.25 * float(bc_action.get("flux_kg_min", 0.0)) + 0.45 * float(ppo_action.get("flux_kg_min", 0.0)) + 0.30 * float(td3_action.get("flux_kg_min", 0.0)),
            "tap_command": bool(bc_action.get("tap_command", False) and ppo_action.get("tap_command", False) and td3_action.get("tap_command", False)),
        }
        return out

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
        ppo_action = self.ppo.act(observation)
        td3_action = self.td3.act(observation)

        fused = self._blend(bc_action, ppo_action, td3_action)
        final_action, reason = self._safety_supervisor(observation, fused)

        self.last_info = {
            "bc_proposal": dict(bc_action),
            "ppo_proposal": dict(ppo_action),
            "td3_proposal": dict(td3_action),
            "hybrid_action": dict(fused),
            "final_action": dict(final_action),
            "safety_reason": reason,
            "pipeline": "state->(bc,ppo,td3)->agentic_fusion->safety_supervisor->final",
        }
        return final_action
