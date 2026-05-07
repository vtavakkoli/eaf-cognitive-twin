from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentAction:
    power_mw: float
    oxygen_nm3_min: float
    ng_nm3_min: float
    carbon_kg_min: float
    flux_kg_min: float
    tap_command: bool = False


@dataclass
class StepResult:
    observation: dict[str, float | str | bool]
    reward: float
    done: bool
    info: dict[str, Any]


ActionDict = dict[str, float | bool]
ObservationDict = dict[str, float | str | bool]
