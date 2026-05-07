from __future__ import annotations

from abc import ABC, abstractmethod

from agents.types import ActionDict, ObservationDict


class BasePolicy(ABC):
    name: str = "base"

    @abstractmethod
    def act(self, observation: ObservationDict) -> ActionDict:
        raise NotImplementedError

    def reset(self) -> None:
        return None
