from __future__ import annotations

from dataclasses import replace

from eaf_twin.config.defaults import default_config
from eaf_twin.config.loader import load_config
from eaf_twin.domain.models import FurnaceConfig
from eaf_twin.models.first_principles import FirstPrinciplesModel
from eaf_twin.simulation.schedule import active_setpoints, stage_name

from agents.safety import SafetyFilter
from agents.types import ActionDict, ObservationDict, StepResult


class EAFController:
    def __init__(self, config: FurnaceConfig, enhanced_model: bool = True):
        self.config = replace(config)
        self.model = FirstPrinciplesModel(self.config, enhanced=enhanced_model)
        self.safety = SafetyFilter()
        self.state = None
        self.prev_action: ActionDict | None = None
        self.warnings: list[str] = []

    @classmethod
    def from_path(cls, config_path=None, enhanced_model: bool = True) -> "EAFController":
        cfg = load_config(config_path) if config_path else default_config()
        return cls(cfg, enhanced_model=enhanced_model)

    def reset(self) -> ObservationDict:
        self.state = self.model.initialize_state()
        self.prev_action = active_setpoints(self.config, 0.0) | {"tap_command": False}
        self.warnings = []
        return self._observation(last_extras={})

    def default_schedule_action(self) -> ActionDict:
        assert self.state is not None
        t_min = self.state.time_s / 60.0
        return active_setpoints(self.config, t_min) | {"tap_command": False}

    def _can_tap(self) -> bool:
        s = self.state
        assert s is not None
        cfg = self.config
        return (
            s.melted_fraction >= 0.95
            and s.liquid_steel_kg >= 0.75 * cfg.tap_target_steel_kg
            and s.steel_temp_k >= cfg.steel_melt_temp_k
        )

    def _observation(self, last_extras: dict) -> ObservationDict:
        s = self.state
        assert s is not None
        return {
            "time_min": s.time_s / 60.0,
            "phase": stage_name(s.time_s / 60.0, s.melted_fraction),
            "bath_temp_c": s.steel_temp_k - 273.15,
            "scrap_temp_c": s.solid_scrap_temp_k - 273.15,
            "slag_temp_c": s.slag_temp_k - 273.15,
            "offgas_temp_c": s.offgas_temp_k - 273.15,
            "melted_fraction": s.melted_fraction,
            "remaining_solid_kg": s.solid_scrap_kg + s.solid_dri_kg,
            "liquid_steel_kg": s.liquid_steel_kg,
            "steel_carbon_wt_pct": s.steel_carbon_wt_pct,
            "cum_electric_mwh": s.cum_electric_j / 3.6e9,
            "cum_oxygen_nm3": s.cum_oxygen_nm3,
            "cum_ng_nm3": s.cum_ng_nm3,
            "cum_carbon_kg": s.cum_carbon_kg,
            "cum_tapped_kg": s.cum_tapped_kg,
            "tapping_started": s.tapping_started,
            "can_tap": self._can_tap(),
            "last_stage": str(last_extras.get("stage", "")),
        }

    def _reward(self) -> float:
        s = self.state
        assert s is not None
        temp_c = s.steel_temp_k - 273.15
        temp_penalty = abs(temp_c - self.config.tap_target_temp_c) / 25.0
        energy_penalty = (s.cum_electric_j / 3.6e9) * 0.002
        progress_bonus = 2.0 * s.melted_fraction + 0.00002 * s.cum_tapped_kg
        carbon_penalty = abs(s.steel_carbon_wt_pct - 0.05) * 2.0
        return progress_bonus - temp_penalty - energy_penalty - carbon_penalty

    def step(self, action: ActionDict) -> StepResult:
        assert self.state is not None, "call reset() before step()"
        dt_min = self.config.dt_s / 60.0
        t_prev = self.state.time_s
        self.model.apply_charge_events(self.state, max(0.0, t_prev - self.config.dt_s), t_prev)
        safe_action = self.safety.apply(action, self.prev_action, dt_min=dt_min, can_tap=self._can_tap())
        extras = self.model._step_dynamics(self.state, safe_action, self.warnings)
        self.model.validate_state(self.state, self.warnings)
        self.state.time_s += self.config.dt_s
        self.prev_action = safe_action

        done = self.state.tap_end_time_s is not None or (self.state.time_s / 60.0) >= self.config.heat_duration_min
        obs = self._observation(extras)
        info = {"warnings": list(self.warnings), "safe_action": safe_action, **extras}
        return StepResult(observation=obs, reward=self._reward(), done=done, info=info)
