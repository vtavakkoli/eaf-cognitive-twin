from __future__ import annotations

from agents.base import BasePolicy
from agents.types import ActionDict, ObservationDict


class LLMPolicy(BasePolicy):
    name = "llm_stub"

    def act(self, observation: ObservationDict) -> ActionDict:
        # Placeholder: keep conservative defaults until an LLM backend is integrated.
        return {"power_mw": 60.0, "oxygen_nm3_min": 40.0, "ng_nm3_min": 8.0, "carbon_kg_min": 8.0, "flux_kg_min": 60.0, "tap_command": False}
