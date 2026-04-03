from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from eaf_twin.config.defaults import default_config
from eaf_twin.domain.models import ChargeEvent, FurnaceConfig, StageWindow


class ConfigError(ValueError):
    pass


def validate_config(cfg: FurnaceConfig) -> None:
    if cfg.dt_s <= 0 or cfg.dt_s > 20:
        raise ConfigError("dt_s must be in (0, 20].")
    if cfg.heat_duration_min <= 0:
        raise ConfigError("heat_duration_min must be positive.")
    if not cfg.stage_windows:
        raise ConfigError("stage_windows cannot be empty.")
    for i, w in enumerate(cfg.stage_windows):
        if w.end_min <= w.start_min:
            raise ConfigError(f"Stage window {i} has end <= start.")
        if min(w.power_mw, w.oxygen_nm3_min, w.ng_nm3_min, w.carbon_kg_min, w.flux_kg_min) < 0:
            raise ConfigError(f"Stage window {i} has negative setpoint.")
    for i in range(1, len(cfg.stage_windows)):
        if cfg.stage_windows[i].start_min < cfg.stage_windows[i - 1].start_min:
            raise ConfigError("stage_windows must be sorted by start_min.")
    for ev in cfg.charge_events:
        if ev.scrap_kg < 0 or ev.dri_kg < 0:
            raise ConfigError("charge event masses must be non-negative.")


def load_config(path: Path | None) -> FurnaceConfig:
    cfg = default_config()
    if path is None:
        validate_config(cfg)
        return cfg
    data = json.loads(path.read_text())
    for key, value in data.items():
        if key == "stage_windows":
            cfg.stage_windows = [StageWindow(**item) for item in value]
        elif key == "charge_events":
            cfg.charge_events = [ChargeEvent(**item) for item in value]
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ConfigError(f"Unknown config field: {key}")
    validate_config(cfg)
    return cfg


def save_config(path: Path, config: FurnaceConfig) -> None:
    path.write_text(json.dumps(asdict(config), indent=2))
