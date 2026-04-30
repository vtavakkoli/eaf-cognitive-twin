from __future__ import annotations

import copy
from dataclasses import replace

from eaf_twin.config.defaults import default_config
from eaf_twin.config.loader import load_config
from eaf_twin.domain.models import FurnaceConfig
from eaf_twin.models.first_principles import FirstPrinciplesModel
from eaf_twin.simulation.schedule import active_setpoints, in_downtime, stage_name

from agents.safety import SafetyFilter
from agents.types import ActionDict, ObservationDict, StepResult


class EAFController:
    TAP_WINDOW_START_MIN = 66.0
    TAP_WINDOW_END_MIN = 70.0

    def __init__(self, config: FurnaceConfig, enhanced_model: bool = True):
        self.config = replace(config)
        self.model = FirstPrinciplesModel(self.config, enhanced=enhanced_model)
        self.model_name = "Model_C_enhanced_hybrid" if enhanced_model else self.model.name
        self.safety = SafetyFilter()
        self.state = None
        self.prev_action: ActionDict | None = None
        self.warnings: list[str] = []
        self._last_cum_electric_mwh = 0.0
        self._last_cum_oxygen_nm3 = 0.0
        self._last_cum_ng_nm3 = 0.0

    @classmethod
    def from_path(cls, config_path=None, enhanced_model: bool = True) -> "EAFController":
        cfg = load_config(config_path) if config_path else default_config()
        return cls(cfg, enhanced_model=enhanced_model)

    def clone_state(self):
        assert self.state is not None
        return copy.deepcopy(self.state)

    def reset(self) -> ObservationDict:
        self.state = self.model.initialize_state()
        self.prev_action = active_setpoints(self.config, 0.0) | {"tap_command": False}
        self.warnings = []
        self._last_cum_electric_mwh = 0.0
        self._last_cum_oxygen_nm3 = 0.0
        self._last_cum_ng_nm3 = 0.0
        return self._observation(last_extras={})

    def default_schedule_action(self) -> ActionDict:
        assert self.state is not None
        t_min = self.state.time_s / 60.0
        return active_setpoints(self.config, t_min) | {"tap_command": False}

    @property
    def tap_window_start_min(self) -> float:
        return min(60.0, self.config.heat_duration_min - 5.0)

    @property
    def tap_window_end_min(self) -> float:
        return self.config.heat_duration_min

    def _can_tap(self) -> bool:
        s = self.state
        assert s is not None
        cfg = self.config
        t_min = s.time_s / 60.0
        return (
            self.tap_window_start_min <= t_min <= self.tap_window_end_min
            and
            s.melted_fraction >= 0.95
            and s.liquid_steel_kg >= 0.75 * cfg.tap_target_steel_kg
            and s.steel_temp_k >= cfg.steel_melt_temp_k
        )

    def _observation(self, last_extras: dict) -> ObservationDict:
        s = self.state
        assert s is not None
        obs = {
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
            "model_name": self.model_name,
            "default_schedule_action": self.default_schedule_action(),
            "_state_obj": copy.deepcopy(s),
            "_config_obj": self.config,
            "_model_obj": self.model,
        }
        obs.update({k: v for k, v in last_extras.items() if k.startswith("reward_") or k.startswith("penalty_") or k == "terminal_reward"})
        return obs

    def _reward_components(self, safety_flags: dict[str, bool | str], done: bool) -> dict[str, float]:
        s = self.state
        assert s is not None
        cfg = self.config
        temp_c = s.steel_temp_k - 273.15
        tap_target = cfg.tap_target_temp_c
        reward_tap_success = 0.0
        reward_mass_quality = 0.0
        reward_temp_quality = 0.0
        cum_electric_mwh = s.cum_electric_j / 3.6e9
        step_electric_mwh = max(0.0, cum_electric_mwh - self._last_cum_electric_mwh)
        step_oxygen_nm3 = max(0.0, s.cum_oxygen_nm3 - self._last_cum_oxygen_nm3)
        step_ng_nm3 = max(0.0, s.cum_ng_nm3 - self._last_cum_ng_nm3)
        penalty_energy = -0.25 * step_electric_mwh
        penalty_oxygen = -0.0006 * step_oxygen_nm3
        penalty_ng = -0.001 * step_ng_nm3
        penalty_carbon = -10.0 * abs(s.steel_carbon_wt_pct - 0.05)
        penalty_temperature_violation = -50.0 if bool(safety_flags.get("temperature_violation", False)) else 0.0
        penalty_invalid_tap = -10.0 if bool(safety_flags.get("invalid_tap_command", False)) else 0.0
        penalty_action_smoothness = 0.0
        final_phase_bonus = 0.0
        final_phase_penalty = 0.0
        if done and s.tap_end_time_s is None:
            if temp_c < cfg.steel_melt_temp_c:
                # Hard failure: episode ends cold at 61 min and cannot tap useful steel.
                final_phase_penalty = -250.0 - 2.0 * (cfg.steel_melt_temp_c - temp_c)
            else:
                # Encourage agents to finish at least above melt temperature even if they miss tap.
                final_phase_bonus = 25.0

        if done and s.tap_end_time_s is not None:
            reward_tap_success = 300.0
            reward_mass_quality = -abs(s.cum_tapped_kg - cfg.tap_target_steel_kg) / 900.0
            reward_temp_quality = -abs(temp_c - tap_target) / 8.0
        terminal_reward = reward_tap_success + reward_mass_quality + reward_temp_quality if s.tap_end_time_s is not None else max(-300.0, -40.0 + final_phase_bonus + final_phase_penalty)
        step_reward = sum(
            [
                reward_tap_success,
                reward_mass_quality,
                reward_temp_quality,
                penalty_energy,
                penalty_oxygen,
                penalty_ng,
                penalty_carbon,
                penalty_temperature_violation,
                penalty_invalid_tap,
                penalty_action_smoothness,
            ]
        )
        raw_reward = step_reward + (terminal_reward if done else 0.0)
        clip_abs = float(getattr(cfg, "rl_reward_clip_abs", 0.0) or 0.0)
        if clip_abs > 0.0:
            raw_reward = float(max(-clip_abs, min(clip_abs, raw_reward)))
        reward_norm = float(getattr(cfg, "rl_reward_normalize_scale", 1.0) or 1.0)
        total_reward = raw_reward / max(1e-9, reward_norm)
        self._last_cum_electric_mwh = cum_electric_mwh
        self._last_cum_oxygen_nm3 = s.cum_oxygen_nm3
        self._last_cum_ng_nm3 = s.cum_ng_nm3
        return {
            "step_reward": step_reward,
            "reward_tap_success": reward_tap_success,
            "reward_mass_quality": reward_mass_quality,
            "reward_temp_quality": reward_temp_quality,
            "penalty_energy": penalty_energy,
            "penalty_oxygen": penalty_oxygen,
            "penalty_ng": penalty_ng,
            "penalty_carbon": penalty_carbon,
            "penalty_temperature_violation": penalty_temperature_violation,
            "penalty_invalid_tap": penalty_invalid_tap,
            "penalty_action_smoothness": penalty_action_smoothness,
            "reward_final_phase_bonus": final_phase_bonus,
            "penalty_final_phase_cold_bath": final_phase_penalty,
            "terminal_reward": terminal_reward if done else 0.0,
            "total_reward": total_reward,
        }

    def step(self, action: ActionDict) -> StepResult:
        assert self.state is not None, "call reset() before step()"
        dt_min = self.config.dt_s / 60.0
        t_prev = self.state.time_s
        self.model.apply_charge_events(self.state, max(0.0, t_prev - self.config.dt_s), t_prev)
        current_obs = self._observation(last_extras={})
        is_down = in_downtime(self.config, t_prev / 60.0)
        safe_action, safety_flags = self.safety.apply(
            action,
            self.prev_action,
            dt_min=dt_min,
            can_tap=self._can_tap(),
            observation=current_obs,
            max_temp_c=self.config.max_temp_c,
            is_downtime=is_down,
        )
        extras = self.model._step_dynamics(self.state, safe_action, self.warnings)
        self.model.validate_state(self.state, self.warnings)
        self.state.time_s += self.config.dt_s
        self.prev_action = safe_action

        done = self.state.tap_end_time_s is not None or (self.state.time_s / 60.0) >= max(self.config.heat_duration_min, self.tap_window_end_min)
        reward_components = self._reward_components(safety_flags, done)
        obs = self._observation({**extras, **reward_components})
        info = {"warnings": list(self.warnings), "safe_action": safe_action, **extras, **safety_flags, "is_downtime": is_down, **reward_components}
        return StepResult(observation=obs, reward=reward_components["total_reward"], done=done, info=info)
