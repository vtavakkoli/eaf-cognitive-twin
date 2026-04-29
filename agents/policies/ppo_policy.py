from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.base import BasePolicy
from agents.policies.rl_common import ACTION_NAMES, normalized_obs_vec, safe_discrete_action
from agents.types import ActionDict, ObservationDict


class PPOPolicy(BasePolicy):
    """Minimal PPO actor-critic policy with softmax actor and linear value head."""

    name = "ppo"

    def __init__(self, actor_w: np.ndarray | None = None, value_w: np.ndarray | None = None):
        self.actor_w = np.array(actor_w, dtype=float) if actor_w is not None else np.zeros((len(ACTION_NAMES), 13), dtype=float)
        self.value_w = np.array(value_w, dtype=float) if value_w is not None else np.zeros(13, dtype=float)

    def _x(self, obs: ObservationDict) -> np.ndarray:
        return np.asarray(normalized_obs_vec(obs), dtype=float)

    def probs(self, obs: ObservationDict) -> np.ndarray:
        logits = self.actor_w @ self._x(obs)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.maximum(exp.sum(), 1e-12)

    def value(self, obs: ObservationDict) -> float:
        return float(self.value_w @ self._x(obs))

    def action_name(self, obs: ObservationDict) -> str:
        return ACTION_NAMES[int(np.argmax(self.probs(obs)))]

    def act(self, observation: ObservationDict) -> ActionDict:
        return safe_discrete_action(self.action_name(observation), observation)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez(f, actor_w=self.actor_w, value_w=self.value_w)

    @classmethod
    def load(cls, path: Path) -> "PPOPolicy":
        load_path = path if path.exists() else Path(f"{path}.npz")
        ckpt = np.load(load_path)
        return cls(actor_w=ckpt["actor_w"], value_w=ckpt["value_w"])
