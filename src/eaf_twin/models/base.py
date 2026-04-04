from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eaf_twin.constants import EPS, J_PER_GJ, J_PER_MWH, SECONDS_PER_MIN
from eaf_twin.domain.models import FurnaceConfig, FurnaceState
from eaf_twin.simulation.schedule import active_setpoints
from eaf_twin.units import clamp
from eaf_twin.validation.checks import validate_state_physics


@dataclass
class ModelResult:
    model_name: str
    scenario_name: str
    df: pd.DataFrame
    summary: dict[str, float]
    warnings: list[str]
    runtime_s: float


class BaseEAFModel:
    name = "Base"

    def __init__(self, config: FurnaceConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def initialize_state(self) -> FurnaceState:
        first_scrap = sum(e.scrap_kg for e in self.config.charge_events if abs(e.time_min) < 1e-9)
        first_dri = sum(e.dri_kg for e in self.config.charge_events if abs(e.time_min) < 1e-9)
        if first_scrap == 0:
            first_scrap = 0.55 * self.config.initial_scrap_kg
        return FurnaceState(
            time_s=0.0,
            solid_scrap_kg=first_scrap,
            solid_dri_kg=first_dri,
            liquid_steel_kg=self.config.initial_hot_heel_kg,
            slag_kg=self.config.initial_slag_kg,
            steel_temp_k=self.config.initial_steel_temp_c + 273.15 if self.config.initial_hot_heel_kg > 0 else self.config.ambient_temp_k,
            slag_temp_k=self.config.initial_slag_temp_c + 273.15 if self.config.initial_slag_kg > 0 else self.config.ambient_temp_k,
            offgas_temp_k=self.config.initial_offgas_temp_c + 273.15,
            steel_carbon_kg=0.006 * max(self.config.initial_hot_heel_kg, 1.0),
            feo_slag_kg=350.0,
        )

    def apply_charge_events(self, state: FurnaceState, t_prev_s: float, t_now_s: float) -> None:
        t_prev_min, t_now_min = t_prev_s / SECONDS_PER_MIN, t_now_s / SECONDS_PER_MIN
        for ev in self.config.charge_events:
            if t_prev_min < ev.time_min <= t_now_min:
                state.solid_scrap_kg += ev.scrap_kg
                state.solid_dri_kg += ev.dri_kg
                shock_k = 10.0 + 0.00008 * (ev.scrap_kg + ev.dri_kg)
                state.steel_temp_k -= shock_k
                state.slag_temp_k -= 0.8 * shock_k

    def validate_state(self, state: FurnaceState, warnings: list[str]) -> None:
        warnings.extend(validate_state_physics(state, self.config.min_temp_k, self.config.max_temp_k))

    def record_row(self, state: FurnaceState, inputs: dict[str, float], extras: dict[str, float]) -> dict[str, float]:
        row = {
            "time_min": state.time_s / SECONDS_PER_MIN,
            **inputs,
            "solid_scrap_kg": state.solid_scrap_kg,
            "solid_dri_kg": state.solid_dri_kg,
            "liquid_steel_kg": state.liquid_steel_kg,
            "cum_tapped_kg": state.cum_tapped_kg,
            "slag_kg": state.slag_kg,
            "steel_temp_k": state.steel_temp_k,
            "slag_temp_k": state.slag_temp_k,
            "offgas_temp_k": state.offgas_temp_k,
            "steel_temp_c": state.steel_temp_k - 273.15,
            "slag_temp_c": state.slag_temp_k - 273.15,
            "offgas_temp_c": state.offgas_temp_k - 273.15,
            "melted_fraction": state.melted_fraction,
            "cum_electric_mwh": state.cum_electric_j / J_PER_MWH,
            "cum_chemical_gj": state.cum_chemical_j / J_PER_GJ,
            "cum_useful_heat_gj": state.cum_useful_heat_j / J_PER_GJ,
            "cum_losses_gj": state.cum_losses_j / J_PER_GJ,
            "cum_oxygen_nm3": state.cum_oxygen_nm3,
            "cum_ng_nm3": state.cum_ng_nm3,
            "cum_carbon_kg": state.cum_carbon_kg,
            "steel_carbon_wt_pct": state.steel_carbon_wt_pct,
            "feo_slag_kg": state.feo_slag_kg,
            "tap_start_min": None if state.tap_start_time_s is None else state.tap_start_time_s / SECONDS_PER_MIN,
            "tap_end_min": None if state.tap_end_time_s is None else state.tap_end_time_s / SECONDS_PER_MIN,
        }
        row.update(extras)
        row["steel_temp_sensor_c"] = row["steel_temp_c"] + self.rng.normal(0, self.config.measurement_noise_std)
        return row

    def compute_summary(self, df: pd.DataFrame, runtime_s: float, warnings: list[str]) -> dict[str, float]:
        final = df.iloc[-1]
        tapped_kg = float(final["cum_tapped_kg"])
        tapped_t = tapped_kg / 1000.0 if tapped_kg > EPS else float("nan")
        return {
            "heat_time_min": float(final["time_min"]),
            "tap_temp_k": float(final["steel_temp_k"]),
            "tap_temp_c": float(final["steel_temp_c"]),
            "cum_tapped_kg": float(final["cum_tapped_kg"]),
            "tap_start_min": float(final["tap_start_min"]) if pd.notna(final["tap_start_min"]) else float("nan"),
            "tap_end_min": float(final["tap_end_min"]) if pd.notna(final["tap_end_min"]) else float("nan"),
            "flat_bath_time_min": float(max(0.0, final["time_min"] - 8.0)),
            "total_electric_mwh": float(final["cum_electric_mwh"]),
            "total_chemical_gj": float(final["cum_chemical_gj"]),
            "total_losses_gj": float(final["cum_losses_gj"]),
            "total_useful_heat_gj": float(final["cum_useful_heat_gj"]),
            "electric_kwh_per_tapped_t": float(final["cum_electric_mwh"] * 1000 / tapped_t) if tapped_t == tapped_t else float("nan"),
            "oxygen_nm3_per_tapped_t": float(final["cum_oxygen_nm3"] / tapped_t) if tapped_t == tapped_t else float("nan"),
            "ng_nm3_per_tapped_t": float(final["cum_ng_nm3"] / tapped_t) if tapped_t == tapped_t else float("nan"),
            "carbon_kg_per_tapped_t": float(final["cum_carbon_kg"] / tapped_t) if tapped_t == tapped_t else float("nan"),
            "final_slag_kg": float(final["slag_kg"]),
            "final_carbon_wt_pct": float(final["steel_carbon_wt_pct"]),
            "warning_count": float(len(warnings)),
            "runtime_s": runtime_s,
        }

    def simulate(self) -> ModelResult:
        raise NotImplementedError

    def run_loop(self, step_fn):
        start = time.perf_counter()
        state = self.initialize_state()
        rows, warnings = [], []
        n_steps = int(self.config.heat_duration_min * SECONDS_PER_MIN / self.config.dt_s)
        for _ in range(n_steps + 1):
            self.apply_charge_events(state, max(0.0, state.time_s - self.config.dt_s), state.time_s)
            inputs = active_setpoints(self.config, state.time_s / SECONDS_PER_MIN)
            extras = step_fn(state, inputs, warnings)
            self.validate_state(state, warnings)
            rows.append(self.record_row(state, inputs, extras))
            state.time_s += self.config.dt_s
            if state.tap_end_time_s is not None:
                break
        df = pd.DataFrame(rows)
        runtime = time.perf_counter() - start
        return ModelResult(self.name, self.config.heat_name, df, self.compute_summary(df, runtime, warnings), warnings, runtime)


def start_or_continue_tapping(state: FurnaceState, cfg: FurnaceConfig) -> float:
    dt = cfg.dt_s
    ready_by_melt = state.melted_fraction >= 0.98 and state.steel_temp_k >= cfg.tap_target_temp_k
    ready_by_time = (state.time_s / SECONDS_PER_MIN) >= 55.0 and state.liquid_steel_kg >= 0.92 * cfg.tap_target_steel_kg
    if ready_by_melt or ready_by_time:
        state.tapping_started = True
        if state.tap_start_time_s is None:
            state.tap_start_time_s = state.time_s
    tap_mass = 0.0
    if state.tapping_started:
        tap_mass = min(state.liquid_steel_kg, cfg.tap_rate_kg_s * dt)
        state.liquid_steel_kg -= tap_mass
        state.cum_tapped_kg += tap_mass
        if state.liquid_steel_kg <= 500.0 and state.tap_end_time_s is None:
            state.tap_end_time_s = state.time_s
    return tap_mass
