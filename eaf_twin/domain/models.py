from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MaterialType(str, Enum):
    SCRAP = "scrap"
    DRI = "dri"
    SLAG = "slag"
    LIQUID_STEEL = "liquid_steel"
    OFF_GAS = "off_gas"


@dataclass
class StageWindow:
    start_min: float
    end_min: float
    power_mw: float
    oxygen_nm3_min: float
    ng_nm3_min: float
    carbon_kg_min: float
    flux_kg_min: float


@dataclass
class ChargeEvent:
    time_min: float
    scrap_kg: float = 0.0
    dri_kg: float = 0.0


@dataclass
class FurnaceConfig:
    heat_name: str = "default_100t_heat"
    tap_target_steel_kg: float = 100_000.0
    initial_scrap_kg: float = 105_000.0
    initial_dri_kg: float = 0.0
    initial_slag_kg: float = 2_500.0

    ambient_temp_c: float = 25.0
    scrap_temp_c: float = 25.0
    initial_steel_temp_c: float = 1550.0
    initial_slag_temp_c: float = 1550.0
    initial_offgas_temp_c: float = 400.0
    tap_target_temp_c: float = 1645.0

    cp_steel_j_kgk: float = 820.0
    cp_slag_j_kgk: float = 1000.0
    cp_scrap_j_kgk: float = 700.0
    cp_dri_j_kgk: float = 680.0
    cp_offgas_j_kgk: float = 1150.0
    latent_heat_steel_j_kg: float = 272_000.0
    steel_melt_temp_c: float = 1535.0

    area_effective_m2: float = 230.0
    ua_wall_w_k: float = 45_000.0
    radiation_loss_factor: float = 0.18

    eta_arc_bore_in: float = 0.70
    eta_arc_melting: float = 0.82
    eta_arc_refining: float = 0.75
    eta_arc_superheat: float = 0.73
    eta_burner: float = 0.62
    oxygen_reaction_efficiency: float = 0.80
    carbon_reaction_efficiency: float = 0.72
    post_combustion_factor: float = 0.28

    lhv_ng_j_nm3: float = 35.8e6
    oxygen_heat_j_nm3: float = 5.4e6
    carbon_heat_j_kg: float = 10.5e6
    fe_oxidation_ratio_per_nm3_o2: float = 1.5
    decarb_kg_per_nm3_o2: float = 0.18
    flux_to_slag_factor: float = 0.98

    slag_to_bath_heat_coeff_w_k: float = 14_000.0
    foamy_slag_loss_reduction: float = 0.20
    dri_reduction_endotherm_j_kg: float = 125_000.0
    dri_fe_metallization: float = 0.92
    max_offgas_temp_c: float = 2200.0

    stage_windows: list[StageWindow] = field(default_factory=list)
    charge_events: list[ChargeEvent] = field(default_factory=list)

    heat_duration_min: float = 65.0
    dt_s: float = 2.0
    max_temp_c: float = 1850.0
    min_temp_c: float = 20.0
    measurement_noise_std: float = 0.0
    random_seed: int = 42

    tap_rate_kg_s: float = 1300.0
    downtime_start_min: Optional[float] = None
    downtime_duration_min: float = 0.0


@dataclass
class FurnaceState:
    time_s: float
    solid_scrap_kg: float
    solid_dri_kg: float
    liquid_steel_kg: float
    slag_kg: float
    steel_temp_c: float
    slag_temp_c: float
    offgas_temp_c: float
    steel_carbon_kg: float
    feo_slag_kg: float
    solid_scrap_temp_c: float

    cum_electric_j: float = 0.0
    cum_chemical_j: float = 0.0
    cum_useful_heat_j: float = 0.0
    cum_losses_j: float = 0.0
    cum_oxygen_nm3: float = 0.0
    cum_ng_nm3: float = 0.0
    cum_carbon_kg: float = 0.0
    cum_tapped_kg: float = 0.0
    tap_start_time_s: Optional[float] = None
    tap_end_time_s: Optional[float] = None
    tapping_started: bool = False

    @property
    def melted_fraction(self) -> float:
        total = self.solid_scrap_kg + self.solid_dri_kg + self.liquid_steel_kg
        return 0.0 if total <= 1e-9 else self.liquid_steel_kg / total

    @property
    def steel_carbon_wt_pct(self) -> float:
        return 100.0 * self.steel_carbon_kg / max(self.liquid_steel_kg, 1e-9)


@dataclass
class ScenarioDefinition:
    name: str
    description: str
    config: FurnaceConfig


@dataclass
class ResultSummary:
    scenario: str
    model: str
    status: str
    heat_time_min: float
    tap_temp_c: float
    cum_tapped_kg: float
    electric_kwh_per_tapped_t: float
    warning_count: int
