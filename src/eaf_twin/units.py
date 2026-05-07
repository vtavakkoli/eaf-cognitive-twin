"""Unit conversion helpers."""


def celsius_to_kelvin(t_c: float) -> float:
    return t_c + 273.15


def kelvin_to_celsius(t_k: float) -> float:
    return t_k - 273.15


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
