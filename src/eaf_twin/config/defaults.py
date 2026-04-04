from __future__ import annotations

from dataclasses import replace

from eaf_twin.domain.models import ChargeEvent, FurnaceConfig, StageWindow


def default_stage_windows() -> list[StageWindow]:
    return [
        StageWindow(0, 8, 45, 35, 22, 10, 80),
        StageWindow(8, 32, 78, 70, 16, 20, 140),
        StageWindow(32, 50, 62, 60, 9, 15, 120),
        StageWindow(50, 60, 50, 40, 4, 6, 60),
        StageWindow(60, 65, 20, 10, 2, 0, 0),
    ]


def default_charge_events(config: FurnaceConfig) -> list[ChargeEvent]:
    first = 0.58 * config.initial_scrap_kg
    second = config.initial_scrap_kg - first
    dri1 = 0.5 * config.initial_dri_kg
    dri2 = config.initial_dri_kg - dri1
    return [
        ChargeEvent(0.0, scrap_kg=first, dri_kg=dri1),
        ChargeEvent(9.0, scrap_kg=second, dri_kg=dri2),
    ]


def default_config() -> FurnaceConfig:
    cfg = FurnaceConfig()
    cfg.stage_windows = default_stage_windows()
    cfg.charge_events = default_charge_events(cfg)
    return cfg


def scenario_configs(base: FurnaceConfig) -> dict[str, FurnaceConfig]:
    scenarios = {"base_case": replace(base)}
    scenarios["higher_oxygen"] = replace(
        base,
        stage_windows=[replace(w, oxygen_nm3_min=w.oxygen_nm3_min * 1.2) for w in base.stage_windows],
    )
    scenarios["higher_natural_gas"] = replace(
        base,
        stage_windows=[replace(w, ng_nm3_min=w.ng_nm3_min * 1.35) for w in base.stage_windows],
    )
    better = replace(base, foamy_slag_loss_reduction=0.30)
    better.eta_arc_melting = min(0.90, base.eta_arc_melting + 0.05)
    better.eta_arc_refining = min(0.86, base.eta_arc_refining + 0.04)
    scenarios["improved_foamy_slag"] = better

    scenarios["dri20"] = replace(base, initial_dri_kg=20_000.0)
    scenarios["delayed_melting_downtime"] = replace(base, downtime_start_min=28.0, downtime_duration_min=6.0)
    return scenarios
