#!/usr/bin/env python3
"""
EAF Heat Simulator (research-oriented baseline)
==============================================

Usage
-----
1) Local Python run:
   python eaf_simulator.py --output-dir outputs --dt 2.0

2) Docker Compose run:
   docker compose up --build full-run

What this script does
---------------------
- Simulates a 100 t scrap-based EAF heat with three model fidelities:
  Model A (empirical), Model B (first-principles lumped), Model C (enhanced hybrid).
- Runs multiple scenarios (base + five variants).
- Exports per-model time series CSV files.
- Exports scenario comparison tables and summary JSON.
- Creates PNG figures for key trajectories and a sensitivity bar chart.

Dependencies
------------
Python 3.12+, numpy, pandas, matplotlib, scipy (optional for future extension).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------
# Constants and helper routines
# -----------------------------

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/m2/K4]
EPS = 1e-9
SECONDS_PER_MIN = 60.0


def celsius_to_kelvin(t_c: float) -> float:
    return t_c + 273.15


def kelvin_to_celsius(t_k: float) -> float:
    return t_k - 273.15


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smooth_step(x: float, x0: float, x1: float) -> float:
    """Return 0..1 smooth transition around [x0, x1]."""
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    z = (x - x0) / (x1 - x0)
    return z * z * (3 - 2 * z)


# -----------------------------
# Configuration data structures
# -----------------------------


@dataclass
class StageWindow:
    """Piecewise-constant setpoint segment for process inputs.

    Times are specified in minutes from heat start.
    """

    start_min: float
    end_min: float
    power_mw: float
    oxygen_nm3_min: float
    ng_nm3_min: float
    carbon_kg_min: float
    flux_kg_min: float


@dataclass
class ChargeEvent:
    """Scrap/DRI/HBI charging event parameters."""

    time_min: float
    scrap_kg: float
    dri_kg: float = 0.0


@dataclass
class FurnaceConfig:
    """Global configuration for physics, operations, numerics, and outputs."""

    # Physical design and capacity
    heat_name: str = "default_100t_heat"
    tap_target_steel_kg: float = 100_000.0
    initial_scrap_kg: float = 105_000.0
    initial_dri_fraction: float = 0.0
    initial_slag_kg: float = 2_500.0

    # Temperatures [degC]
    ambient_temp_c: float = 25.0
    scrap_temp_c: float = 25.0
    initial_steel_temp_c: float = 1550.0
    initial_slag_temp_c: float = 1550.0
    initial_offgas_temp_c: float = 400.0
    tap_target_temp_c: float = 1645.0

    # Material and thermophysical properties
    cp_steel_j_kgk: float = 820.0
    cp_slag_j_kgk: float = 1000.0
    cp_scrap_j_kgk: float = 700.0
    cp_offgas_j_kgk: float = 1150.0
    latent_heat_steel_j_kg: float = 272_000.0
    steel_melt_temp_c: float = 1535.0
    furnace_internal_thermal_mass_j_k: float = 15e6

    # Heat transfer and losses
    area_effective_m2: float = 230.0
    ua_wall_w_k: float = 45_000.0
    radiation_loss_factor: float = 0.18

    # Energy conversion
    eta_arc_bore_in: float = 0.70
    eta_arc_melting: float = 0.82
    eta_arc_refining: float = 0.75
    eta_arc_superheat: float = 0.73
    eta_burner: float = 0.62
    oxygen_reaction_efficiency: float = 0.80
    carbon_reaction_efficiency: float = 0.72
    post_combustion_factor: float = 0.28

    # Chemical properties and rates
    lhv_ng_j_nm3: float = 35.8e6
    oxygen_heat_j_nm3: float = 5.4e6
    carbon_heat_j_kg: float = 10.5e6
    fe_oxidation_ratio_per_nm3_o2: float = 1.5  # kg Fe oxidized per Nm3 O2
    decarb_kg_per_nm3_o2: float = 0.18
    flux_to_slag_factor: float = 0.98

    # Process dynamics
    slag_to_bath_heat_coeff_w_k: float = 14_000.0
    foamy_slag_loss_reduction: float = 0.20
    dri_extra_heat_j_kg: float = 110_000.0
    max_offgas_temp_c: float = 2200.0

    # Schedule and events
    stage_windows: List[StageWindow] = field(default_factory=list)
    charge_events: List[ChargeEvent] = field(default_factory=list)

    # Numerics
    heat_duration_min: float = 65.0
    dt_s: float = 2.0
    max_temp_c: float = 1850.0
    min_temp_c: float = 20.0
    measurement_noise_std: float = 0.0
    random_seed: int = 42

    # Tapping and downtime
    tap_rate_kg_s: float = 1300.0
    downtime_start_min: Optional[float] = None
    downtime_duration_min: float = 0.0


@dataclass
class FurnaceState:
    """Dynamic state of the furnace for model B/C (and partly A)."""

    time_s: float
    solid_scrap_kg: float
    liquid_steel_kg: float
    slag_kg: float
    steel_temp_c: float
    slag_temp_c: float
    offgas_temp_c: float
    steel_carbon_kg: float
    feo_slag_kg: float

    cum_electric_j: float = 0.0
    cum_chemical_j: float = 0.0
    cum_oxygen_nm3: float = 0.0
    cum_ng_nm3: float = 0.0
    cum_carbon_kg: float = 0.0
    cum_losses_j: float = 0.0
    tapping_started: bool = False

    @property
    def melted_fraction(self) -> float:
        total_metal = self.solid_scrap_kg + self.liquid_steel_kg
        return 0.0 if total_metal <= EPS else self.liquid_steel_kg / total_metal

    @property
    def steel_carbon_wt_pct(self) -> float:
        return 100.0 * self.steel_carbon_kg / max(self.liquid_steel_kg, EPS)


@dataclass
class ModelResult:
    """Container for one model/scenario run outputs."""

    model_name: str
    scenario_name: str
    df: pd.DataFrame
    summary: Dict[str, float]
    warnings: List[str]
    runtime_s: float


# -----------------------------
# Default schedule / scenarios
# -----------------------------


def default_stage_windows() -> List[StageWindow]:
    """Default stage profile for a representative 100 t EAF heat."""
    return [
        StageWindow(0, 8, power_mw=45, oxygen_nm3_min=35, ng_nm3_min=22, carbon_kg_min=10, flux_kg_min=80),  # bore-in
        StageWindow(8, 32, power_mw=78, oxygen_nm3_min=70, ng_nm3_min=16, carbon_kg_min=20, flux_kg_min=140),  # main melt
        StageWindow(32, 50, power_mw=62, oxygen_nm3_min=60, ng_nm3_min=9, carbon_kg_min=15, flux_kg_min=120),  # refining
        StageWindow(50, 60, power_mw=50, oxygen_nm3_min=40, ng_nm3_min=4, carbon_kg_min=6, flux_kg_min=60),  # superheat
        StageWindow(60, 65, power_mw=20, oxygen_nm3_min=10, ng_nm3_min=2, carbon_kg_min=0, flux_kg_min=0),  # tapping prep
    ]


def default_charge_events(config: FurnaceConfig) -> List[ChargeEvent]:
    """Two-bucket charging style event schedule."""
    first_bucket = 0.58 * config.initial_scrap_kg
    second_bucket = config.initial_scrap_kg - first_bucket
    return [
        ChargeEvent(time_min=0.0, scrap_kg=first_bucket),
        ChargeEvent(time_min=9.0, scrap_kg=second_bucket),
    ]


def load_config(path: Optional[Path]) -> FurnaceConfig:
    """Load configuration JSON if provided, else return default baseline config."""
    cfg = FurnaceConfig()
    cfg.stage_windows = default_stage_windows()
    cfg.charge_events = default_charge_events(cfg)
    if path is None:
        return cfg

    data = json.loads(path.read_text())
    for key, value in data.items():
        if key == "stage_windows":
            cfg.stage_windows = [StageWindow(**item) for item in value]
        elif key == "charge_events":
            cfg.charge_events = [ChargeEvent(**item) for item in value]
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def save_config(path: Path, config: FurnaceConfig) -> None:
    """Save current config to JSON (for reproducibility and parameter sweeps)."""
    payload = asdict(config)
    path.write_text(json.dumps(payload, indent=2))


def scenario_configs(base: FurnaceConfig) -> Dict[str, FurnaceConfig]:
    """Create required scenario set from baseline config."""
    scenarios: Dict[str, FurnaceConfig] = {"base_case": replace(base)}

    high_o2 = replace(base)
    high_o2.stage_windows = [
        replace(w, oxygen_nm3_min=w.oxygen_nm3_min * 1.20) for w in base.stage_windows
    ]
    scenarios["higher_oxygen"] = high_o2

    high_ng = replace(base)
    high_ng.stage_windows = [replace(w, ng_nm3_min=w.ng_nm3_min * 1.35) for w in base.stage_windows]
    scenarios["higher_natural_gas"] = high_ng

    better_foam = replace(base, foamy_slag_loss_reduction=0.30)
    better_foam.eta_arc_melting = min(0.90, base.eta_arc_melting + 0.05)
    better_foam.eta_arc_refining = min(0.86, base.eta_arc_refining + 0.04)
    scenarios["improved_foamy_slag"] = better_foam

    dri_20 = replace(base, initial_dri_fraction=0.20)
    scenarios["dri20"] = dri_20

    delayed = replace(base, downtime_start_min=28.0, downtime_duration_min=6.0)
    scenarios["delayed_melting_downtime"] = delayed

    return scenarios


# -----------------------------
# Schedule helpers
# -----------------------------


def in_downtime(config: FurnaceConfig, t_min: float) -> bool:
    """Return whether downtime interruption is active at time t."""
    if config.downtime_start_min is None or config.downtime_duration_min <= 0:
        return False
    return config.downtime_start_min <= t_min < (config.downtime_start_min + config.downtime_duration_min)


def active_setpoints(config: FurnaceConfig, t_min: float) -> Dict[str, float]:
    """Get piecewise setpoints at time t, including downtime overrides."""
    chosen = config.stage_windows[-1]
    for w in config.stage_windows:
        if w.start_min <= t_min < w.end_min:
            chosen = w
            break

    if in_downtime(config, t_min):
        return {
            "power_mw": 0.0,
            "oxygen_nm3_min": 5.0,
            "ng_nm3_min": 1.0,
            "carbon_kg_min": 0.0,
            "flux_kg_min": 0.0,
        }

    return {
        "power_mw": chosen.power_mw,
        "oxygen_nm3_min": chosen.oxygen_nm3_min,
        "ng_nm3_min": chosen.ng_nm3_min,
        "carbon_kg_min": chosen.carbon_kg_min,
        "flux_kg_min": chosen.flux_kg_min,
    }


def stage_name(t_min: float, melted_fraction: float) -> str:
    """Heuristic stage detection by time and melt progress."""
    if t_min < 8:
        return "bore_in"
    if melted_fraction < 0.70:
        return "main_melting"
    if melted_fraction < 0.98:
        return "refining"
    if t_min < 60:
        return "superheat"
    return "tapping"


# -----------------------------
# Model infrastructure
# -----------------------------


class BaseEAFModel:
    """Common interface for all model fidelities."""

    name = "Base"

    def __init__(self, config: FurnaceConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def initialize_state(self) -> FurnaceState:
        """Build initial state including first bucket logic."""
        first_charge = 0.0
        dri_charge = 0.0
        for event in self.config.charge_events:
            if abs(event.time_min) < 1e-9:
                first_charge += event.scrap_kg
                dri_charge += event.dri_kg
        if first_charge <= 0:
            first_charge = 0.55 * self.config.initial_scrap_kg

        init_solid = first_charge + dri_charge
        init_liquid = 8_000.0
        init_carbon = 0.006 * init_liquid

        return FurnaceState(
            time_s=0.0,
            solid_scrap_kg=init_solid,
            liquid_steel_kg=init_liquid,
            slag_kg=self.config.initial_slag_kg,
            steel_temp_c=self.config.initial_steel_temp_c,
            slag_temp_c=self.config.initial_slag_temp_c,
            offgas_temp_c=self.config.initial_offgas_temp_c,
            steel_carbon_kg=init_carbon,
            feo_slag_kg=350.0,
        )

    def arc_efficiency(self, stage: str, melted_fraction: float, foamy_factor: float) -> float:
        """Stage/melt dependent arc transfer efficiency."""
        cfg = self.config
        base = {
            "bore_in": cfg.eta_arc_bore_in,
            "main_melting": cfg.eta_arc_melting,
            "refining": cfg.eta_arc_refining,
            "superheat": cfg.eta_arc_superheat,
            "tapping": 0.40,
        }[stage]
        progress_gain = 0.04 * smooth_step(melted_fraction, 0.20, 0.95)
        eta = base + progress_gain + 0.05 * foamy_factor
        return clamp(eta, 0.40, 0.92)

    def apply_charge_events(self, state: FurnaceState, t_prev_s: float, t_now_s: float) -> None:
        """Inject bucket events that occur within the current time step."""
        t_prev_min = t_prev_s / SECONDS_PER_MIN
        t_now_min = t_now_s / SECONDS_PER_MIN
        for ev in self.config.charge_events:
            if t_prev_min < ev.time_min <= t_now_min:
                state.solid_scrap_kg += ev.scrap_kg + ev.dri_kg
                # Cold charge cools bath/slag instantly in this lumped approximation.
                thermal_shock = 10.0 + 0.00008 * ev.scrap_kg
                state.steel_temp_c = max(self.config.min_temp_c, state.steel_temp_c - thermal_shock)
                state.slag_temp_c = max(self.config.min_temp_c, state.slag_temp_c - 0.8 * thermal_shock)

    def validate_state(self, state: FurnaceState, warnings: List[str]) -> None:
        """Numerical/physical guardrails with clamping and warnings."""
        for mass_attr in ("solid_scrap_kg", "liquid_steel_kg", "slag_kg", "steel_carbon_kg", "feo_slag_kg"):
            value = getattr(state, mass_attr)
            if value < -1e-6:
                warnings.append(f"Negative mass detected: {mass_attr}={value:.3f}; clamped to 0.")
            setattr(state, mass_attr, max(0.0, value))

        for temp_attr in ("steel_temp_c", "slag_temp_c", "offgas_temp_c"):
            value = getattr(state, temp_attr)
            if not np.isfinite(value):
                warnings.append(f"Non-finite temperature in {temp_attr}; reset to ambient.")
                value = self.config.ambient_temp_c
            if value < self.config.min_temp_c or value > self.config.max_temp_c + 350:
                warnings.append(f"Temperature {temp_attr} out-of-range ({value:.1f} C); clamped.")
            setattr(state, temp_attr, clamp(value, self.config.min_temp_c, self.config.max_temp_c + 350.0))

    def record_row(self, state: FurnaceState, inputs: Dict[str, float], extras: Dict[str, float]) -> Dict[str, float]:
        """Map current state and derived terms to a tabular row."""
        row = {
            "time_min": state.time_s / SECONDS_PER_MIN,
            "power_mw": inputs["power_mw"],
            "oxygen_nm3_min": inputs["oxygen_nm3_min"],
            "ng_nm3_min": inputs["ng_nm3_min"],
            "carbon_kg_min": inputs["carbon_kg_min"],
            "flux_kg_min": inputs["flux_kg_min"],
            "solid_scrap_kg": state.solid_scrap_kg,
            "liquid_steel_kg": state.liquid_steel_kg,
            "slag_kg": state.slag_kg,
            "steel_temp_c": state.steel_temp_c,
            "slag_temp_c": state.slag_temp_c,
            "offgas_temp_c": state.offgas_temp_c,
            "melted_fraction": state.melted_fraction,
            "cum_electric_mwh": state.cum_electric_j / 3.6e9,
            "cum_chemical_gj": state.cum_chemical_j / 1e9,
            "cum_oxygen_nm3": state.cum_oxygen_nm3,
            "cum_ng_nm3": state.cum_ng_nm3,
            "cum_carbon_kg": state.cum_carbon_kg,
            "steel_carbon_wt_pct": state.steel_carbon_wt_pct,
            "feo_slag_kg": state.feo_slag_kg,
            "cum_losses_gj": state.cum_losses_j / 1e9,
        }
        row.update(extras)

        if self.config.measurement_noise_std > 0:
            noise = self.config.measurement_noise_std
            row["steel_temp_sensor_c"] = row["steel_temp_c"] + self.rng.normal(0, noise)
            row["offgas_temp_sensor_c"] = row["offgas_temp_c"] + self.rng.normal(0, noise * 2)
        else:
            row["steel_temp_sensor_c"] = row["steel_temp_c"]
            row["offgas_temp_sensor_c"] = row["offgas_temp_c"]
        return row

    def compute_summary(self, df: pd.DataFrame, runtime_s: float, warnings: List[str]) -> Dict[str, float]:
        """Calculate key KPIs for comparisons and plausibility checks."""
        final = df.iloc[-1]
        steel_t = max(final["liquid_steel_kg"], EPS) / 1000.0
        elec_kwh_t = final["cum_electric_mwh"] * 1000.0 / steel_t
        oxy_nm3_t = final["cum_oxygen_nm3"] / steel_t
        ng_nm3_t = final["cum_ng_nm3"] / steel_t
        carbon_kg_t = final["cum_carbon_kg"] / steel_t

        useful_in = final.get("cum_useful_heat_gj", final["cum_chemical_gj"]) + (final["cum_electric_mwh"] * 3.6)
        eff = useful_in / max(useful_in + final["cum_losses_gj"], EPS)

        summary = {
            "heat_time_min": float(final["time_min"]),
            "tap_temp_c": float(final["steel_temp_c"]),
            "tap_steel_kg": float(final["liquid_steel_kg"]),
            "final_slag_kg": float(final["slag_kg"]),
            "melted_fraction": float(final["melted_fraction"]),
            "electric_kwh_t": float(elec_kwh_t),
            "oxygen_nm3_t": float(oxy_nm3_t),
            "ng_nm3_t": float(ng_nm3_t),
            "carbon_kg_t": float(carbon_kg_t),
            "thermal_efficiency_est": float(clamp(eff, 0.0, 1.0)),
            "runtime_s": float(runtime_s),
            "warning_count": float(len(warnings)),
        }
        return summary

    def simulate(self) -> ModelResult:
        raise NotImplementedError


# -----------------------------
# Model A: Empirical reduced model
# -----------------------------


class EmpiricalModel(BaseEAFModel):
    """Low-order empirical dynamic model for rapid sensitivity studies."""

    name = "Model_A_empirical"

    def simulate(self) -> ModelResult:
        start = time.perf_counter()
        state = self.initialize_state()
        cfg = self.config
        warnings: List[str] = []
        rows: List[Dict[str, float]] = []

        n_steps = int(cfg.heat_duration_min * SECONDS_PER_MIN / cfg.dt_s)
        for _ in range(n_steps + 1):
            t_min = state.time_s / SECONDS_PER_MIN
            inputs = active_setpoints(cfg, t_min)
            stg = stage_name(t_min, state.melted_fraction)

            self.apply_charge_events(state, state.time_s - cfg.dt_s, state.time_s)
            dt = cfg.dt_s

            power_w = inputs["power_mw"] * 1e6
            q_elec = power_w * dt
            q_burner = cfg.eta_burner * inputs["ng_nm3_min"] / 60.0 * cfg.lhv_ng_j_nm3 * dt
            q_oxy = 0.62 * cfg.oxygen_reaction_efficiency * inputs["oxygen_nm3_min"] / 60.0 * cfg.oxygen_heat_j_nm3 * dt

            eta_stage = {
                "bore_in": 0.58,
                "main_melting": 0.72,
                "refining": 0.63,
                "superheat": 0.60,
                "tapping": 0.40,
            }[stg]
            useful = eta_stage * q_elec + 0.75 * q_burner + q_oxy

            t_internal_k = celsius_to_kelvin((state.steel_temp_c + state.slag_temp_c) * 0.5)
            t_amb_k = celsius_to_kelvin(cfg.ambient_temp_c)
            q_loss = cfg.ua_wall_w_k * (t_internal_k - t_amb_k) * dt
            q_loss += cfg.radiation_loss_factor * SIGMA * cfg.area_effective_m2 * (t_internal_k**4 - t_amb_k**4) * dt
            q_loss = max(0.0, q_loss)

            q_net = useful - q_loss
            q_need_per_kg = cfg.cp_scrap_j_kgk * (cfg.steel_melt_temp_c - cfg.scrap_temp_c) + cfg.latent_heat_steel_j_kg
            max_melt = max(0.0, q_net / max(q_need_per_kg, EPS))
            kin_melt = (inputs["power_mw"] * 2.6 + inputs["oxygen_nm3_min"] * 0.55 + inputs["ng_nm3_min"] * 0.30)
            melt_rate = min(max_melt / dt, kin_melt, state.solid_scrap_kg / dt)

            m_melt = max(0.0, melt_rate * dt)
            state.solid_scrap_kg -= m_melt
            state.liquid_steel_kg += m_melt

            cpi = cfg.cp_steel_j_kgk * max(state.liquid_steel_kg, 10_000.0)
            dT = (q_net - m_melt * cfg.latent_heat_steel_j_kg) / max(cpi, EPS)
            state.steel_temp_c += dT
            state.slag_temp_c += 0.35 * dT
            state.offgas_temp_c = clamp(
                cfg.ambient_temp_c + 0.12 * (state.steel_temp_c - cfg.ambient_temp_c) + inputs["oxygen_nm3_min"] * 4.0,
                cfg.ambient_temp_c,
                cfg.max_offgas_temp_c,
            )

            # Simple mass side terms
            state.slag_kg += inputs["flux_kg_min"] / 60.0 * dt * cfg.flux_to_slag_factor
            fe_loss = min(state.liquid_steel_kg * 0.00003 * dt, state.liquid_steel_kg * 0.002)
            state.liquid_steel_kg -= fe_loss
            state.feo_slag_kg += fe_loss

            decarb = min(state.steel_carbon_kg, inputs["oxygen_nm3_min"] / 60.0 * dt * cfg.decarb_kg_per_nm3_o2 * 0.7)
            inj_c = inputs["carbon_kg_min"] / 60.0 * dt
            state.steel_carbon_kg += inj_c - decarb

            if state.melted_fraction > 0.995 and state.steel_temp_c >= cfg.tap_target_temp_c:
                state.tapping_started = True
            if state.tapping_started:
                tap_m = min(state.liquid_steel_kg, cfg.tap_rate_kg_s * dt)
                state.liquid_steel_kg -= tap_m

            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_burner + q_oxy
            state.cum_oxygen_nm3 += inputs["oxygen_nm3_min"] / 60.0 * dt
            state.cum_ng_nm3 += inputs["ng_nm3_min"] / 60.0 * dt
            state.cum_carbon_kg += inputs["carbon_kg_min"] / 60.0 * dt
            state.cum_losses_j += q_loss

            extras = {
                "stage": stg,
                "q_useful_mw": useful / dt / 1e6,
                "q_loss_mw": q_loss / dt / 1e6,
                "q_melt_mw": m_melt * cfg.latent_heat_steel_j_kg / dt / 1e6,
                "offgas_co_frac": 0.42,
                "offgas_co2_frac": 0.24,
                "offgas_o2_frac": 0.03,
                "offgas_n2_frac": 0.31,
                "cum_useful_heat_gj": (state.cum_electric_j * eta_stage + state.cum_chemical_j) / 1e9,
            }
            rows.append(self.record_row(state, inputs, extras))

            self.validate_state(state, warnings)
            state.time_s += dt

            if state.tapping_started and state.liquid_steel_kg <= 500.0:
                break

        df = pd.DataFrame(rows)
        runtime = time.perf_counter() - start
        summary = self.compute_summary(df, runtime, warnings)
        return ModelResult(self.name, cfg.heat_name, df, summary, warnings, runtime)


# -----------------------------
# Model B/C common first-principles implementation
# -----------------------------


class FirstPrinciplesModel(BaseEAFModel):
    """Dynamic mass-energy model with explicit steel/slag/off-gas balances."""

    name = "Model_B_first_principles"

    def __init__(self, config: FurnaceConfig, enhanced: bool = False):
        super().__init__(config)
        self.enhanced = enhanced
        if enhanced:
            self.name = "Model_C_enhanced_hybrid"

    def compute_foamy_factor(self, state: FurnaceState, inputs: Dict[str, float], stg: str) -> float:
        """Estimate slag foaming factor (0..1) from C/O ratio and slag state."""
        base = 0.25 if stg in ("bore_in", "main_melting") else 0.45
        if not self.enhanced:
            return base
        o2 = inputs["oxygen_nm3_min"]
        c = inputs["carbon_kg_min"]
        ratio = c / max(o2 * 0.18, 1e-3)
        feo_idx = state.feo_slag_kg / max(state.slag_kg, 1.0)
        foam = base + 0.25 * math.tanh(1.6 * (ratio - 1.0)) + 0.20 * smooth_step(feo_idx, 0.08, 0.22)
        return clamp(foam, 0.05, 0.95)

    def split_reaction_heat(self, total_reaction_q: float, foamy_factor: float) -> Tuple[float, float, float]:
        """Split reaction heat to steel, slag, and gas sinks."""
        if not self.enhanced:
            return 0.58 * total_reaction_q, 0.14 * total_reaction_q, 0.28 * total_reaction_q
        steel_share = 0.54 + 0.12 * foamy_factor
        slag_share = 0.18 + 0.05 * foamy_factor
        gas_share = max(0.08, 1.0 - steel_share - slag_share)
        return steel_share * total_reaction_q, slag_share * total_reaction_q, gas_share * total_reaction_q

    def correction_term(self, state: FurnaceState, t_min: float) -> float:
        """Optional simple data-driven correction (W) for model C temperature bias."""
        if not self.enhanced:
            return 0.0
        # Lightweight regression-style correction based on residual trends vs stage.
        trend = 0.0
        trend += 2.6e6 * smooth_step(state.melted_fraction, 0.5, 0.95)
        trend -= 1.1e6 * smooth_step(t_min, 52.0, 64.0)
        return trend

    def step(self, state: FurnaceState, inputs: Dict[str, float], warnings: List[str]) -> Dict[str, float]:
        """Advance state by one explicit Euler step using coupled balances."""
        cfg = self.config
        dt = cfg.dt_s
        t_min = state.time_s / SECONDS_PER_MIN
        stg = stage_name(t_min, state.melted_fraction)

        power_w = inputs["power_mw"] * 1e6
        o2_flow = inputs["oxygen_nm3_min"] / 60.0
        ng_flow = inputs["ng_nm3_min"] / 60.0
        c_flow = inputs["carbon_kg_min"] / 60.0
        flux_flow = inputs["flux_kg_min"] / 60.0

        foamy = self.compute_foamy_factor(state, inputs, stg)
        eta_arc = self.arc_efficiency(stg, state.melted_fraction, foamy)
        q_elec_useful = eta_arc * power_w * dt

        eta_burn = cfg.eta_burner if not self.enhanced else cfg.eta_burner * (0.95 + 0.08 * foamy)
        q_burner = eta_burn * ng_flow * cfg.lhv_ng_j_nm3 * dt

        # Oxygen/C chemistry terms.
        q_oxy_total = o2_flow * cfg.oxygen_heat_j_nm3 * cfg.oxygen_reaction_efficiency * dt
        decarb_rate = o2_flow * cfg.decarb_kg_per_nm3_o2 * (0.65 + 0.25 * smooth_step(state.melted_fraction, 0.7, 1.0))
        decarb_mass = min(state.steel_carbon_kg, max(0.0, decarb_rate * dt))
        q_c_chem = c_flow * cfg.carbon_heat_j_kg * cfg.carbon_reaction_efficiency * dt

        q_rxn_total = q_oxy_total + q_c_chem
        q_rxn_steel, q_rxn_slag, q_rxn_gas = self.split_reaction_heat(q_rxn_total, foamy)

        # Fe oxidation and slag growth.
        fe_oxid = o2_flow * cfg.fe_oxidation_ratio_per_nm3_o2 * (0.75 if stg == "main_melting" else 0.55)
        fe_oxid_m = min(state.liquid_steel_kg * 0.0015, max(0.0, fe_oxid * dt))
        oxide_gen = fe_oxid_m * 1.29

        # Scrap/DRI melt from available net heat.
        q_need_scrap = cfg.cp_scrap_j_kgk * (cfg.steel_melt_temp_c - cfg.scrap_temp_c) + cfg.latent_heat_steel_j_kg
        if cfg.initial_dri_fraction > 0:
            q_need_scrap += cfg.initial_dri_fraction * cfg.dri_extra_heat_j_kg

        # Coupled losses (wall + radiation, reduced by foamy slag for model C)
        t_internal_k = celsius_to_kelvin(0.65 * state.steel_temp_c + 0.35 * state.slag_temp_c)
        t_amb_k = celsius_to_kelvin(cfg.ambient_temp_c)
        q_loss_conv = cfg.ua_wall_w_k * (t_internal_k - t_amb_k) * dt
        rad_factor = cfg.radiation_loss_factor * (1.0 - cfg.foamy_slag_loss_reduction * foamy)
        q_loss_rad = rad_factor * SIGMA * cfg.area_effective_m2 * (t_internal_k**4 - t_amb_k**4) * dt
        q_heat_loss = max(0.0, q_loss_conv + max(0.0, q_loss_rad))

        offgas_mass_flow = 1.25 * o2_flow + 0.78 * ng_flow + 0.4 * c_flow + 2.0
        t_gas_k = celsius_to_kelvin(state.offgas_temp_c)
        q_offgas_sens = offgas_mass_flow * cfg.cp_offgas_j_kgk * max(0.0, t_gas_k - t_amb_k) * dt * 0.16

        q_arc_and_chem = q_elec_useful + q_burner + q_rxn_steel
        q_available_for_melting = max(0.0, q_arc_and_chem - 0.30 * q_heat_loss)
        max_melt_energy = q_available_for_melting / max(q_need_scrap, EPS)

        kinetic_limiter = (
            2.4 * inputs["power_mw"]
            + 0.50 * inputs["oxygen_nm3_min"]
            + 0.25 * inputs["ng_nm3_min"]
        )
        if self.enhanced:
            kinetic_limiter *= 1.0 + 0.10 * foamy
            kinetic_limiter *= 1.0 - 0.12 * cfg.initial_dri_fraction

        melt_rate = min(state.solid_scrap_kg / dt, kinetic_limiter, max_melt_energy / dt)
        melt_mass = max(0.0, melt_rate * dt)
        q_melting = melt_mass * q_need_scrap

        # Steel + slag thermal coupling
        q_slag_to_bath = cfg.slag_to_bath_heat_coeff_w_k * (state.slag_temp_c - state.steel_temp_c) * dt

        # Data-driven correction for model C
        q_corr = self.correction_term(state, t_min) * dt

        # Steel bath energy balance
        steel_heat_cap = max(cfg.cp_steel_j_kgk * max(state.liquid_steel_kg, 12_000.0), EPS)
        q_steel_net = (
            q_elec_useful
            + 0.55 * q_burner
            + q_rxn_steel
            + q_slag_to_bath
            + q_corr
            - q_melting
            - 0.62 * q_heat_loss
            - 0.45 * q_offgas_sens
        )
        dT_steel = q_steel_net / steel_heat_cap

        # Slag energy balance
        slag_heat_cap = max(cfg.cp_slag_j_kgk * max(state.slag_kg, 4_000.0), EPS)
        q_slag_net = q_rxn_slag + 0.20 * q_burner - q_slag_to_bath - 0.25 * q_heat_loss
        dT_slag = q_slag_net / slag_heat_cap

        # Offgas temperature dynamic (first-order explicit)
        q_to_gas = q_rxn_gas + 0.25 * q_burner + 0.38 * q_heat_loss
        gas_cap = max(offgas_mass_flow * cfg.cp_offgas_j_kgk * dt + 2.5e5, EPS)
        dT_gas = (q_to_gas - q_offgas_sens) / gas_cap

        # Mass updates
        state.solid_scrap_kg -= melt_mass
        state.liquid_steel_kg += melt_mass - fe_oxid_m
        state.slag_kg += flux_flow * dt * cfg.flux_to_slag_factor + oxide_gen
        state.feo_slag_kg += oxide_gen - 0.06 * state.feo_slag_kg * dt / 60.0

        inj_c = c_flow * dt
        dissolved_loss = 0.0006 * state.steel_carbon_kg * dt
        state.steel_carbon_kg += inj_c - decarb_mass - dissolved_loss

        # Tapping logic
        if state.melted_fraction > 0.997 and state.steel_temp_c >= cfg.tap_target_temp_c:
            state.tapping_started = True
        tap_m = 0.0
        if state.tapping_started:
            tap_m = min(state.liquid_steel_kg, cfg.tap_rate_kg_s * dt)
            state.liquid_steel_kg -= tap_m

        # Apply temperature updates after mass update
        state.steel_temp_c += dT_steel
        state.slag_temp_c += dT_slag
        state.offgas_temp_c += dT_gas

        state.offgas_temp_c = clamp(state.offgas_temp_c, cfg.ambient_temp_c, cfg.max_offgas_temp_c)

        state.cum_electric_j += power_w * dt
        state.cum_chemical_j += q_burner + q_oxy_total + q_c_chem
        state.cum_oxygen_nm3 += o2_flow * dt
        state.cum_ng_nm3 += ng_flow * dt
        state.cum_carbon_kg += inj_c
        state.cum_losses_j += q_heat_loss + q_offgas_sens

        # Simplified off-gas composition based on decarb/O2 balance
        o2_in = o2_flow * dt
        o2_for_c = min(o2_in * 0.68, decarb_mass / 0.375)  # approximate stoich split
        co2_frac = clamp(0.12 + 0.40 * cfg.post_combustion_factor, 0.08, 0.45)
        co_frac = clamp(0.48 - 0.35 * cfg.post_combustion_factor, 0.10, 0.70)
        o2_frac = clamp(0.02 + 0.20 * max(0.0, o2_in - o2_for_c) / max(o2_in + 1.0, 1.0), 0.01, 0.20)
        n2_frac = clamp(1.0 - co_frac - co2_frac - o2_frac, 0.20, 0.70)

        extras = {
            "stage": stg,
            "foamy_factor": foamy,
            "eta_arc": eta_arc,
            "q_useful_mw": (q_elec_useful + q_burner + q_rxn_steel) / dt / 1e6,
            "q_loss_mw": (q_heat_loss + q_offgas_sens) / dt / 1e6,
            "q_melt_mw": q_melting / dt / 1e6,
            "q_offgas_mw": q_offgas_sens / dt / 1e6,
            "decarb_kg_s": decarb_mass / dt,
            "tapped_kg_s": tap_m / dt,
            "offgas_co_frac": co_frac,
            "offgas_co2_frac": co2_frac,
            "offgas_o2_frac": o2_frac,
            "offgas_n2_frac": n2_frac,
            "cum_useful_heat_gj": (
                (state.cum_electric_j * eta_arc) + state.cum_chemical_j * 0.72
            )
            / 1e9,
        }
        self.validate_state(state, warnings)
        return extras

    def simulate(self) -> ModelResult:
        start = time.perf_counter()
        state = self.initialize_state()
        cfg = self.config
        warnings: List[str] = []
        rows: List[Dict[str, float]] = []

        n_steps = int(cfg.heat_duration_min * SECONDS_PER_MIN / cfg.dt_s)
        for _ in range(n_steps + 1):
            t_prev = max(0.0, state.time_s - cfg.dt_s)
            self.apply_charge_events(state, t_prev, state.time_s)
            t_min = state.time_s / SECONDS_PER_MIN
            inputs = active_setpoints(cfg, t_min)
            extras = self.step(state, inputs, warnings)
            rows.append(self.record_row(state, inputs, extras))
            state.time_s += cfg.dt_s
            if state.tapping_started and state.liquid_steel_kg <= 500.0:
                break

        df = pd.DataFrame(rows)
        runtime = time.perf_counter() - start
        summary = self.compute_summary(df, runtime, warnings)
        return ModelResult(self.name, cfg.heat_name, df, summary, warnings, runtime)


# -----------------------------
# Validation and reporting
# -----------------------------


def plausibility_checks(result: ModelResult) -> List[str]:
    """Internal validation checks for realistic EAF operating envelopes."""
    issues: List[str] = []
    s = result.summary
    if not (380.0 <= s["electric_kwh_t"] <= 600.0):
        issues.append("Specific electricity outside typical 380-600 kWh/t")
    if not (20.0 <= s["oxygen_nm3_t"] <= 55.0):
        issues.append("Oxygen outside typical 20-55 Nm3/t")
    if not (3.0 <= s["ng_nm3_t"] <= 15.0):
        issues.append("Natural gas outside typical 3-15 Nm3/t")
    if not (1590.0 <= s["tap_temp_c"] <= 1690.0):
        issues.append("Tap temperature outside plausible 1590-1690 C")
    if s["melted_fraction"] < 0.98:
        issues.append("Melted fraction did not reach 0.98")
    if s["tap_steel_kg"] < 92_000.0:
        issues.append("Tapped steel mass too low")
    if s["final_slag_kg"] < 6_000.0 or s["final_slag_kg"] > 20_000.0:
        issues.append("Final slag mass outside plausible 6-20 t")
    return issues


def save_time_series(result: ModelResult, out_dir: Path) -> Path:
    """Save model trajectory to CSV."""
    fname = f"timeseries_{result.scenario_name}_{result.model_name}.csv"
    path = out_dir / fname
    result.df.to_csv(path, index=False)
    return path


def save_summary_table(summaries: List[Dict[str, float]], out_dir: Path, name: str) -> Tuple[Path, Path]:
    """Save summary comparison in CSV and JSON formats."""
    df = pd.DataFrame(summaries)
    csv_path = out_dir / f"summary_{name}.csv"
    json_path = out_dir / f"summary_{name}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    return csv_path, json_path


def plot_model_outputs(result: ModelResult, out_dir: Path) -> List[Path]:
    """Create required PNG plots for a single model run."""
    df = result.df
    t = df["time_min"]
    prefix = f"{result.scenario_name}_{result.model_name}"
    paths: List[Path] = []

    def save(fig_name: str) -> Path:
        path = out_dir / f"plot_{prefix}_{fig_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return path

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["steel_temp_c"], label="Steel")
    plt.plot(t, df["slag_temp_c"], label="Slag")
    plt.plot(t, df["offgas_temp_c"], label="Off-gas")
    plt.xlabel("Time [min]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Temperature trajectories ({prefix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths.append(save("temperatures"))

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["melted_fraction"], color="tab:purple")
    plt.xlabel("Time [min]")
    plt.ylabel("Melted fraction [-]")
    plt.title(f"Melted fraction ({prefix})")
    plt.grid(True, alpha=0.3)
    paths.append(save("melted_fraction"))

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["solid_scrap_kg"], label="Solid scrap")
    plt.plot(t, df["liquid_steel_kg"], label="Liquid steel")
    plt.xlabel("Time [min]")
    plt.ylabel("Mass [kg]")
    plt.title(f"Metal phase masses ({prefix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths.append(save("metal_masses"))

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["cum_electric_mwh"], label="Electric [MWh]")
    plt.plot(t, df["cum_chemical_gj"] / 3.6, label="Chemical [MWh eq]")
    plt.xlabel("Time [min]")
    plt.ylabel("Cumulative energy [MWh]")
    plt.title(f"Cumulative energies ({prefix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths.append(save("cumulative_energy"))

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["cum_oxygen_nm3"], label="Oxygen [Nm3]")
    plt.plot(t, df["cum_ng_nm3"], label="Natural gas [Nm3]")
    plt.plot(t, df["cum_carbon_kg"], label="Carbon [kg]")
    plt.xlabel("Time [min]")
    plt.ylabel("Cumulative consumption")
    plt.title(f"Cumulative consumables ({prefix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths.append(save("consumables"))

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["steel_carbon_wt_pct"], color="tab:brown")
    plt.xlabel("Time [min]")
    plt.ylabel("Steel carbon [wt%]")
    plt.title(f"Carbon trajectory ({prefix})")
    plt.grid(True, alpha=0.3)
    paths.append(save("steel_carbon"))

    plt.figure(figsize=(9, 5))
    plt.stackplot(
        t,
        df["q_useful_mw"].clip(lower=0),
        df["q_melt_mw"].clip(lower=0),
        df["q_loss_mw"].clip(lower=0),
        labels=["Useful in", "Melting sink", "Losses"],
        alpha=0.85,
    )
    plt.xlabel("Time [min]")
    plt.ylabel("Power [MW]")
    plt.title(f"Heat flow stack ({prefix})")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    paths.append(save("heat_stack"))

    return paths


def compare_models_plot(scenario: str, results: List[ModelResult], out_dir: Path) -> Path:
    """Plot model comparison for steel temperature in one scenario."""
    plt.figure(figsize=(9, 5))
    for r in results:
        plt.plot(r.df["time_min"], r.df["steel_temp_c"], label=r.model_name)
    plt.xlabel("Time [min]")
    plt.ylabel("Steel temperature [°C]")
    plt.title(f"Model comparison: steel temperature ({scenario})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = out_dir / f"plot_compare_models_{scenario}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return path


def run_sensitivity(base: FurnaceConfig, out_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """One-factor-at-a-time sensitivity on key efficiency/loss parameters."""
    factors = {
        "eta_arc_melting": [0.9, 1.1],
        "eta_burner": [0.9, 1.1],
        "oxygen_reaction_efficiency": [0.9, 1.1],
        "ua_wall_w_k": [0.85, 1.15],
    }

    baseline_model = FirstPrinciplesModel(replace(base), enhanced=True)
    base_result = baseline_model.simulate()
    base_temp = base_result.summary["tap_temp_c"]
    base_elec = base_result.summary["electric_kwh_t"]

    rows = []
    for pname, multipliers in factors.items():
        deltas = []
        for m in multipliers:
            cfg = replace(base)
            setattr(cfg, pname, getattr(base, pname) * m)
            res = FirstPrinciplesModel(cfg, enhanced=True).simulate()
            delta = abs(res.summary["tap_temp_c"] - base_temp) + abs(res.summary["electric_kwh_t"] - base_elec)
            deltas.append(delta)
        rows.append({"parameter": pname, "sensitivity_index": max(deltas)})

    sens_df = pd.DataFrame(rows).sort_values("sensitivity_index", ascending=True)

    plt.figure(figsize=(8.5, 4.8))
    plt.barh(sens_df["parameter"], sens_df["sensitivity_index"], color="tab:teal")
    plt.xlabel("|ΔTapTemp| + |ΔElec kWh/t|")
    plt.title("Sensitivity ranking (enhanced model)")
    plt.grid(True, alpha=0.3, axis="x")
    sens_plot = out_dir / "plot_sensitivity_ranking.png"
    plt.tight_layout()
    plt.savefig(sens_plot, dpi=140)
    plt.close()

    sens_df.to_csv(out_dir / "sensitivity_table.csv", index=False)
    return sens_df, sens_plot


# -----------------------------
# Main run pipeline
# -----------------------------


def run_full_simulation(config: FurnaceConfig, output_dir: Path) -> None:
    """Run all scenarios and model hierarchies, writing files and summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(output_dir / "resolved_config.json", config)

    scenarios = scenario_configs(config)
    all_summary_rows: List[Dict[str, float]] = []

    for scen_name, scen_cfg in scenarios.items():
        scen_cfg.heat_name = scen_name
        model_runs: List[ModelResult] = []

        for model in (EmpiricalModel(scen_cfg), FirstPrinciplesModel(scen_cfg, False), FirstPrinciplesModel(scen_cfg, True)):
            result = model.simulate()
            result.scenario_name = scen_name

            issues = plausibility_checks(result)
            for issue in issues:
                result.warnings.append(f"plausibility:{issue}")

            save_time_series(result, output_dir)
            plot_model_outputs(result, output_dir)

            row = {
                "scenario": scen_name,
                "model": result.model_name,
                **result.summary,
                "status": "ok" if len(issues) == 0 else "warning",
                "issues": " | ".join(issues),
            }
            all_summary_rows.append(row)
            model_runs.append(result)

        compare_models_plot(scen_name, model_runs, output_dir)

    save_summary_table(all_summary_rows, output_dir, "all_scenarios")
    run_sensitivity(config, output_dir)

    summary_df = pd.DataFrame(all_summary_rows)
    print("\n=== EAF Simulation Completed ===")
    print(summary_df[["scenario", "model", "tap_temp_c", "electric_kwh_t", "oxygen_nm3_t", "ng_nm3_t", "status"]].to_string(index=False))
    print(f"\nOutput directory: {output_dir.resolve()}")


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for simulation execution."""
    parser = argparse.ArgumentParser(description="Dynamic simulator for scrap-based EAF steelmaking heat")
    parser.add_argument("--config", type=Path, default=None, help="Path to optional JSON config")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for CSV/JSON/PNG results")
    parser.add_argument("--dt", type=float, default=None, help="Override time step [s]")
    parser.add_argument("--noise", type=float, default=None, help="Measurement noise stdev [°C]")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for measurement noise")
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    config = load_config(args.config)

    if args.dt is not None:
        if args.dt <= 0 or args.dt > 20:
            raise ValueError("dt must be within (0, 20] seconds")
        config.dt_s = args.dt
    if args.noise is not None:
        if args.noise < 0 or args.noise > 30:
            raise ValueError("noise must be within [0, 30] °C")
        config.measurement_noise_std = args.noise
    if args.seed is not None:
        config.random_seed = args.seed

    if config.dt_s > 5:
        print("WARNING: large dt can reduce numerical fidelity; recommended 1-5 s.")

    run_full_simulation(config, args.output_dir)


if __name__ == "__main__":
    main()
