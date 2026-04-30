from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from agents.base import BasePolicy
from agents.policies.rl_common import ACTION_NAMES, normalized_obs_vec, safe_discrete_action
from agents.types import ActionDict, ObservationDict


class DQNQNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNPolicy(BasePolicy):
    """Neural Double-DQN policy over discrete action set."""

    name = "neural_double_dqn"

    def __init__(self, q_network: DQNQNetwork | None = None):
        self.input_dim = 13
        self.output_dim = len(ACTION_NAMES)
        self.q_network = q_network or DQNQNetwork(self.input_dim, self.output_dim)

    def q_values(self, obs: ObservationDict) -> np.ndarray:
        x = torch.tensor(np.asarray(normalized_obs_vec(obs), dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.q_network(x).squeeze(0).cpu().numpy()
        return q

    def action_name(self, obs: ObservationDict) -> str:
        return ACTION_NAMES[int(np.argmax(self.q_values(obs)))]

    def act(self, observation: ObservationDict) -> ActionDict:
        return safe_discrete_action(self.action_name(observation), observation)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        p = path if path.suffix == ".pt" else path.with_suffix(".pt")
        torch.save({"state_dict": self.q_network.state_dict(), "input_dim": self.input_dim, "output_dim": self.output_dim}, p)

    @classmethod
    def load(cls, path: Path) -> "DQNPolicy":
        p = path if path.suffix == ".pt" else path.with_suffix(".pt")
        ckpt = torch.load(p, map_location="cpu")
        net = DQNQNetwork(int(ckpt.get("input_dim", 13)), int(ckpt.get("output_dim", len(ACTION_NAMES))))
        net.load_state_dict(ckpt["state_dict"])
        return cls(q_network=net)
