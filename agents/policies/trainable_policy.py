from __future__ import annotations

import json
from pathlib import Path

from agents.policies.heuristic import HeuristicParams, HeuristicPolicy


class TrainablePolicy(HeuristicPolicy):
    name: str = "trained"

    def __init__(self, params: HeuristicParams | None = None):
        super().__init__(params=params)

    @classmethod
    def load(cls, path: Path) -> "TrainablePolicy":
        data = json.loads(path.read_text())
        return cls(params=HeuristicParams(**data["params"]))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"policy": self.name, "params": self.to_dict()}, indent=2))
