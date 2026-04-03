"""Calibration scaffolding for future parameter estimation workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationTarget:
    metric_name: str
    observed_value: float
    weight: float = 1.0
