from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.base import BasePolicy
from agents.policies.rl_common import ACTION_NAMES, normalized_obs_vec, safe_discrete_action
from agents.types import ActionDict, ObservationDict


class DQNPolicy(BasePolicy):
    """Lightweight linear Double-DQN style policy over discrete action set."""

    name = "dqn"

    def __init__(self, weights: np.ndarray | None = None):
        self.weights = np.array(weights, dtype=float) if weights is not None else np.zeros((len(ACTION_NAMES), 13), dtype=float)

    def q_values(self, obs: ObservationDict) -> np.ndarray:
        x = np.asarray(normalized_obs_vec(obs), dtype=float)
        return self.weights @ x

    def action_name(self, obs: ObservationDict) -> str:
        return ACTION_NAMES[int(np.argmax(self.q_values(obs)))]

    def act(self, observation: ObservationDict) -> ActionDict:
        return safe_discrete_action(self.action_name(observation), observation)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.weights)

    @classmethod
    def load(cls, path: Path) -> "DQNPolicy":
        p = path
        if p.suffix != ".npy":
            p = p.with_suffix(".npy")
        return cls(weights=np.load(p))
