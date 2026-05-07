from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents.base import BasePolicy
from agents.policies.rl_common import ACTION_NAMES, Discretizer, safe_discrete_action
from agents.types import ActionDict, ObservationDict


class QLearningPolicy(BasePolicy):
    """Tabular/discretized Q-learning controller for EAF actions."""

    name = "q_learning"

    def __init__(self, q_table: dict[str, list[float]] | None = None, epsilon: float = 0.0, seed: int = 7):
        self.discretizer = Discretizer()
        self.q_table = q_table or {}
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    def _ensure_state(self, key: str) -> list[float]:
        if key not in self.q_table:
            self.q_table[key] = [0.0 for _ in ACTION_NAMES]
        return self.q_table[key]

    def act(self, observation: ObservationDict) -> ActionDict:
        key = self.discretizer.encode(observation)
        qvals = self._ensure_state(key)
        if self.rng.random() < self.epsilon:
            a_idx = int(self.rng.integers(0, len(ACTION_NAMES)))
        else:
            a_idx = int(np.argmax(qvals))
        return safe_discrete_action(ACTION_NAMES[a_idx], observation)

    def greedy_action_name(self, observation: ObservationDict) -> str:
        key = self.discretizer.encode(observation)
        qvals = self._ensure_state(key)
        return ACTION_NAMES[int(np.argmax(qvals))]

    def update(self, obs: ObservationDict, action_name: str, reward: float, next_obs: ObservationDict, done: bool, alpha: float, gamma: float) -> None:
        s = self.discretizer.encode(obs)
        ns = self.discretizer.encode(next_obs)
        q = self._ensure_state(s)
        qn = self._ensure_state(ns)
        a = ACTION_NAMES.index(action_name)
        target = float(reward) + (0.0 if done else float(gamma) * float(np.max(qn)))
        q[a] = q[a] + float(alpha) * (target - q[a])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"action_names": ACTION_NAMES, "q_table": self.q_table}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "QLearningPolicy":
        data = json.loads(path.read_text())
        return cls(q_table=data.get("q_table", {}), epsilon=0.0)
