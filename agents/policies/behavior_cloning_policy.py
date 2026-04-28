from __future__ import annotations

import json
from pathlib import Path

from agents.base import BasePolicy
from agents.policies.rl_common import ACTION_NAMES, Discretizer, safe_discrete_action
from agents.types import ActionDict, ObservationDict


class BehaviorCloningPolicy(BasePolicy):
    name = "behavior_cloning"

    def __init__(self, mapping: dict[str, str] | None = None):
        self.discretizer = Discretizer()
        self.mapping = mapping or {}

    def act(self, observation: ObservationDict) -> ActionDict:
        key = self.discretizer.encode(observation)
        return safe_discrete_action(self.mapping.get(key, "medium_power"), observation)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.mapping, indent=2))

    @classmethod
    def load(cls, path: Path) -> "BehaviorCloningPolicy":
        return cls(mapping=json.loads(path.read_text()))
