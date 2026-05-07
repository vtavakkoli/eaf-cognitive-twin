from __future__ import annotations

from pathlib import Path

from agents.base import BasePolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.td3_inspired_policy import TD3InspiredPolicy
from agents.types import ActionDict, ObservationDict


class SafePPOAgenticTD3Policy(BasePolicy):
    """Hybrid policy: PPO proposal blended with TD3-inspired action smoothing and safety gating."""

    name = "safe_ppo_agentic_td3"

    def __init__(self, ppo_policy: PPOPolicy | None = None):
        self.ppo = ppo_policy or PPOPolicy()
        self.td3 = TD3InspiredPolicy()
        self.last_info: dict[str, object] = {}

    @classmethod
    def load(cls, path: Path) -> "SafePPOAgenticTD3Policy":
        return cls(ppo_policy=PPOPolicy.load(path))

    def act(self, observation: ObservationDict) -> ActionDict:
        ppo_action = self.ppo.act(observation)
        td3_action = self.td3.act(observation)
        can_tap = bool(observation.get("can_tap", False))
        is_downtime = bool(observation.get("is_downtime", False))

        blend = {
            "power_mw": 0.65 * float(ppo_action.get("power_mw", 0.0)) + 0.35 * float(td3_action.get("power_mw", 0.0)),
            "oxygen_nm3_min": 0.65 * float(ppo_action.get("oxygen_nm3_min", 0.0)) + 0.35 * float(td3_action.get("oxygen_nm3_min", 0.0)),
            "ng_nm3_min": 0.65 * float(ppo_action.get("ng_nm3_min", 0.0)) + 0.35 * float(td3_action.get("ng_nm3_min", 0.0)),
            "carbon_kg_min": 0.65 * float(ppo_action.get("carbon_kg_min", 0.0)) + 0.35 * float(td3_action.get("carbon_kg_min", 0.0)),
            "flux_kg_min": 0.65 * float(ppo_action.get("flux_kg_min", 0.0)) + 0.35 * float(td3_action.get("flux_kg_min", 0.0)),
            "tap_command": bool(ppo_action.get("tap_command", False) and td3_action.get("tap_command", False)),
        }

        if not can_tap:
            blend["tap_command"] = False
        if is_downtime:
            blend.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0, "tap_command": False})

        self.last_info = {"ppo_raw_action": ppo_action, "td3_raw_action": td3_action, "hybrid_action": blend}
        return blend
