from __future__ import annotations

from agents.base import BasePolicy
from agents.policies.heuristic import HeuristicPolicy
from agents.policies.llm_policy import LLMPolicy
from agents.types import ActionDict, ObservationDict


class HybridPolicy(BasePolicy):
    name = "hybrid_stub"

    def __init__(self) -> None:
        self.heuristic = HeuristicPolicy()
        self.llm = LLMPolicy()

    def act(self, observation: ObservationDict) -> ActionDict:
        act = self.heuristic.act(observation)
        if float(observation["melted_fraction"]) > 0.97:
            llm_act = self.llm.act(observation)
            act["oxygen_nm3_min"] = 0.5 * (act["oxygen_nm3_min"] + llm_act["oxygen_nm3_min"])
        return act
