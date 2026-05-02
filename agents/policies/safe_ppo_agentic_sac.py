from __future__ import annotations

from pathlib import Path

from agents.base import BasePolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.sac_inspired_policy import SACInspiredPolicy
from agents.types import ActionDict, ObservationDict


class SafePPOAgenticSACPolicy(BasePolicy):
    """Hybrid policy: PPO proposal blended with SAC-inspired continuous refinement and safety gating."""

    name = "safe_ppo_agentic_sac"

    def __init__(self, ppo_policy: PPOPolicy | None = None):
        self.ppo = ppo_policy or PPOPolicy()
        self.sac = SACInspiredPolicy()
        self.last_info: dict[str, object] = {}

    @classmethod
    def load(cls, path: Path) -> "SafePPOAgenticSACPolicy":
        return cls(ppo_policy=PPOPolicy.load(path))

    def act(self, observation: ObservationDict) -> ActionDict:
        ppo_action = self.ppo.act(observation)
        sac_action = self.sac.act(observation)
        can_tap = bool(observation.get("can_tap", False))
        is_downtime = bool(observation.get("is_downtime", False))

        blend = {
            "power_mw": 0.6 * float(ppo_action.get("power_mw", 0.0)) + 0.4 * float(sac_action.get("power_mw", 0.0)),
            "oxygen_nm3_min": 0.6 * float(ppo_action.get("oxygen_nm3_min", 0.0)) + 0.4 * float(sac_action.get("oxygen_nm3_min", 0.0)),
            "ng_nm3_min": 0.6 * float(ppo_action.get("ng_nm3_min", 0.0)) + 0.4 * float(sac_action.get("ng_nm3_min", 0.0)),
            "carbon_kg_min": 0.6 * float(ppo_action.get("carbon_kg_min", 0.0)) + 0.4 * float(sac_action.get("carbon_kg_min", 0.0)),
            "flux_kg_min": 0.6 * float(ppo_action.get("flux_kg_min", 0.0)) + 0.4 * float(sac_action.get("flux_kg_min", 0.0)),
            "tap_command": bool(ppo_action.get("tap_command", False) or sac_action.get("tap_command", False)),
        }

        if not can_tap:
            blend["tap_command"] = False
        if is_downtime:
            blend.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0, "tap_command": False})

        self.last_info = {"ppo_raw_action": ppo_action, "sac_raw_action": sac_action, "hybrid_action": blend}
        return blend
