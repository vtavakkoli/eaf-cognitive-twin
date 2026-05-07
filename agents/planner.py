from __future__ import annotations


def choose_target_phase(melted_fraction: float) -> str:
    if melted_fraction < 0.6:
        return "melting"
    if melted_fraction < 0.95:
        return "refining"
    return "tapping"
