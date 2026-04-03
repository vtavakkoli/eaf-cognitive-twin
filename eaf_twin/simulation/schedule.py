from __future__ import annotations

from eaf_twin.domain.models import FurnaceConfig


def smooth_step(x: float, x0: float, x1: float) -> float:
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    z = (x - x0) / (x1 - x0)
    return z * z * (3 - 2 * z)


def in_downtime(config: FurnaceConfig, t_min: float) -> bool:
    if config.downtime_start_min is None or config.downtime_duration_min <= 0:
        return False
    return config.downtime_start_min <= t_min < (config.downtime_start_min + config.downtime_duration_min)


def active_setpoints(config: FurnaceConfig, t_min: float) -> dict[str, float]:
    chosen = config.stage_windows[-1]
    for w in config.stage_windows:
        if w.start_min <= t_min < w.end_min:
            chosen = w
            break
    if in_downtime(config, t_min):
        return {"power_mw": 0.0, "oxygen_nm3_min": 5.0, "ng_nm3_min": 1.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0}
    return {
        "power_mw": chosen.power_mw,
        "oxygen_nm3_min": chosen.oxygen_nm3_min,
        "ng_nm3_min": chosen.ng_nm3_min,
        "carbon_kg_min": chosen.carbon_kg_min,
        "flux_kg_min": chosen.flux_kg_min,
    }


def stage_name(t_min: float, melted_fraction: float) -> str:
    if t_min < 8:
        return "bore_in"
    if melted_fraction < 0.70:
        return "main_melting"
    if melted_fraction < 0.98:
        return "refining"
    if t_min < 60:
        return "superheat"
    return "tapping"
