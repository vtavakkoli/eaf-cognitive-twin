from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMemory:
    observations: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, observation: dict, action: dict, reward: float) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
