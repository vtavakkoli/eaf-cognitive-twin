from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from agents.base import BasePolicy
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.rl_common import ACTION_NAMES, safe_discrete_action, normalized_obs_vec
from agents.policies.td3_inspired_policy import TD3InspiredPolicy
from agents.types import ActionDict, ObservationDict


ACTION_VECTOR_DIM = 6
SETPOINT_VECTOR_DIM = 6
TD3BC_GOAL_VECTOR_DIM = 6
GOAL_ERROR_VECTOR_DIM = 6
PHASE_VECTOR_DIM = 5
STATE_VECTOR_DIM = 13
PREDICTOR_INPUT_DIM = STATE_VECTOR_DIM + ACTION_VECTOR_DIM + SETPOINT_VECTOR_DIM + TD3BC_GOAL_VECTOR_DIM + GOAL_ERROR_VECTOR_DIM + PHASE_VECTOR_DIM + 1
FEATURE_DIM = STATE_VECTOR_DIM + ACTION_VECTOR_DIM + SETPOINT_VECTOR_DIM + TD3BC_GOAL_VECTOR_DIM + GOAL_ERROR_VECTOR_DIM + PHASE_VECTOR_DIM + STATE_VECTOR_DIM + 1


def _clip(value: float, lo: float = -1.5, hi: float = 1.5) -> float:
    return float(min(hi, max(lo, value)))


def _cfg_value(obs: ObservationDict, name: str, default: float) -> float:
    cfg = obs.get("_config_obj")
    return float(getattr(cfg, name, default)) if cfg is not None else float(default)


def action_to_vec(action: ActionDict | None) -> np.ndarray:
    """Normalize the previous multimodal EAF control action."""
    if not action:
        return np.zeros(ACTION_VECTOR_DIM, dtype=float)
    return np.asarray(
        [
            _clip(float(action.get("power_mw", 0.0)) / 120.0, 0.0, 1.5),
            _clip(float(action.get("oxygen_nm3_min", 0.0)) / 100.0, 0.0, 1.5),
            _clip(float(action.get("ng_nm3_min", 0.0)) / 30.0, 0.0, 1.5),
            _clip(float(action.get("carbon_kg_min", 0.0)) / 25.0, 0.0, 1.5),
            _clip(float(action.get("flux_kg_min", 0.0)) / 180.0, 0.0, 1.5),
            float(bool(action.get("tap_command", False))),
        ],
        dtype=float,
    )


def td3bc_goal_action(
    obs: ObservationDict,
    bc_policy: BehaviorCloningPolicy | None = None,
    td3_policy: TD3InspiredPolicy | None = None,
) -> ActionDict:
    """Create an intermediate goal proposal from TD3 and behavior cloning.

    The goal is not the final executed action. It is a short-horizon operating
    target used by the JEPA predictor and PPO actor. TD3 gives smooth continuous
    process targets, while BC anchors the target toward expert-like operation.
    """
    bc = bc_policy or BehaviorCloningPolicy()
    td3 = td3_policy or TD3InspiredPolicy()
    bc_action = bc.act(obs)
    td3_action = td3.act(obs)

    blended = {
        "power_mw": 0.40 * float(bc_action.get("power_mw", 0.0)) + 0.60 * float(td3_action.get("power_mw", 0.0)),
        "oxygen_nm3_min": 0.40 * float(bc_action.get("oxygen_nm3_min", 0.0)) + 0.60 * float(td3_action.get("oxygen_nm3_min", 0.0)),
        "ng_nm3_min": 0.40 * float(bc_action.get("ng_nm3_min", 0.0)) + 0.60 * float(td3_action.get("ng_nm3_min", 0.0)),
        "carbon_kg_min": 0.40 * float(bc_action.get("carbon_kg_min", 0.0)) + 0.60 * float(td3_action.get("carbon_kg_min", 0.0)),
        "flux_kg_min": 0.40 * float(bc_action.get("flux_kg_min", 0.0)) + 0.60 * float(td3_action.get("flux_kg_min", 0.0)),
        "tap_command": bool(bc_action.get("tap_command", False) and td3_action.get("tap_command", False)),
    }

    if bool(obs.get("is_downtime", False)):
        blended.update({"power_mw": 0.0, "oxygen_nm3_min": 0.0, "ng_nm3_min": 0.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0, "tap_command": False})
    if not bool(obs.get("can_tap", False)):
        blended["tap_command"] = False
    return blended


def td3bc_goal_vec(
    obs: ObservationDict,
    bc_policy: BehaviorCloningPolicy | None = None,
    td3_policy: TD3InspiredPolicy | None = None,
) -> np.ndarray:
    """Encode the TD3+BC short-horizon goal proposal."""
    return action_to_vec(td3bc_goal_action(obs, bc_policy=bc_policy, td3_policy=td3_policy))


def setpoint_vec(obs: ObservationDict) -> np.ndarray:
    """Encode operation recipe/set-points as a goal vector.

    This is intentionally derived from the simulator configuration so the policy
    can condition its action on the desired endpoint, not only on the current
    furnace state.
    """
    target_temp_c = _cfg_value(obs, "tap_target_temp_c", 1640.0)
    melt_temp_c = _cfg_value(obs, "steel_melt_temp_c", 1600.0)
    target_mass_kg = _cfg_value(obs, "tap_target_steel_kg", 100000.0)
    max_temp_c = _cfg_value(obs, "max_temp_c", 1700.0)
    heat_duration_min = _cfg_value(obs, "heat_duration_min", 70.0)
    return np.asarray(
        [
            _clip((target_temp_c - 1200.0) / 700.0, 0.0, 1.5),
            _clip((melt_temp_c - 1200.0) / 700.0, 0.0, 1.5),
            _clip(target_mass_kg / 120000.0, 0.0, 1.5),
            _clip(0.05 / 0.20, 0.0, 1.5),  # target endpoint carbon wt pct, normalized
            _clip((max_temp_c - 1200.0) / 700.0, 0.0, 1.5),
            _clip(heat_duration_min / 90.0, 0.0, 1.5),
        ],
        dtype=float,
    )


def goal_error_vec(obs: ObservationDict) -> np.ndarray:
    """Measure the gap between current furnace state and operation set-points."""
    target_temp_c = _cfg_value(obs, "tap_target_temp_c", 1640.0)
    target_mass_kg = _cfg_value(obs, "tap_target_steel_kg", 100000.0)
    max_temp_c = _cfg_value(obs, "max_temp_c", 1700.0)
    heat_duration_min = _cfg_value(obs, "heat_duration_min", 70.0)

    bath_temp_c = float(obs.get("bath_temp_c", 20.0))
    liquid_steel_kg = float(obs.get("liquid_steel_kg", 0.0))
    melted_fraction = float(obs.get("melted_fraction", 0.0))
    carbon = float(obs.get("steel_carbon_wt_pct", 0.05))
    time_min = float(obs.get("time_min", 0.0))

    return np.asarray(
        [
            _clip((target_temp_c - bath_temp_c) / 300.0),
            _clip(1.0 - melted_fraction, 0.0, 1.0),
            _clip((target_mass_kg - liquid_steel_kg) / max(target_mass_kg, 1e-9)),
            _clip((0.05 - carbon) / 0.05),
            _clip((max_temp_c - bath_temp_c) / 250.0),
            _clip(time_min / max(heat_duration_min, 1e-9), 0.0, 1.5),
        ],
        dtype=float,
    )


def phase_vec(obs: ObservationDict) -> np.ndarray:
    """Encode coarse EAF process phase as a positional/process embedding."""
    phase = str(obs.get("phase", "")).lower()
    time_min = float(obs.get("time_min", 0.0))
    melted = float(obs.get("melted_fraction", 0.0))
    can_tap = bool(obs.get("can_tap", False))
    downtime = bool(obs.get("is_downtime", False))

    if downtime:
        idx = 4
    elif can_tap:
        idx = 3
    elif "ref" in phase or melted >= 0.90 or time_min >= 50.0:
        idx = 2
    elif "melt" in phase or melted > 0.05:
        idx = 1
    else:
        idx = 0
    out = np.zeros(PHASE_VECTOR_DIM, dtype=float)
    out[idx] = 1.0
    return out


def predictor_input(
    obs: ObservationDict,
    previous_action: ActionDict | None,
    bc_policy: BehaviorCloningPolicy | None = None,
    td3_policy: TD3InspiredPolicy | None = None,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(normalized_obs_vec(obs), dtype=float),
            action_to_vec(previous_action),
            setpoint_vec(obs),
            td3bc_goal_vec(obs, bc_policy=bc_policy, td3_policy=td3_policy),
            goal_error_vec(obs),
            phase_vec(obs),
            np.ones(1, dtype=float),
        ]
    )


class GoalConditionedJEPAPPOPolicy(BasePolicy):
    """Goal-conditioned JEPA-PPO controller with TD3+BC goal setting.

    The policy adds three inputs that plain PPO does not explicitly model:
    1. previous multimodal action embedding,
    2. operation set-point/recipe embedding,
    3. TD3+BC short-horizon goal proposal,
    4. JEPA-style latent prediction of the next furnace state.

    TD3+BC sets the intermediate operating goal because PPO-SafeAgent-TD3BC
    and Behavior Cloning are the strongest prior controllers in this benchmark.
    PPO still chooses the discrete action. The auxiliary latent predictor is
    trained from observed transitions with a representation prediction loss.
    """

    name = "goal_conditioned_jepa_ppo"

    def __init__(
        self,
        actor_w: np.ndarray | None = None,
        value_w: np.ndarray | None = None,
        predictor_w: np.ndarray | None = None,
        bc_policy: BehaviorCloningPolicy | None = None,
        td3_policy: TD3InspiredPolicy | None = None,
    ):
        self.actor_w = np.asarray(actor_w, dtype=float) if actor_w is not None else np.zeros((len(ACTION_NAMES), FEATURE_DIM), dtype=float)
        self.value_w = np.asarray(value_w, dtype=float) if value_w is not None else np.zeros(FEATURE_DIM, dtype=float)
        self.predictor_w = np.asarray(predictor_w, dtype=float) if predictor_w is not None else np.zeros((STATE_VECTOR_DIM, PREDICTOR_INPUT_DIM), dtype=float)
        self.bc_policy = bc_policy or BehaviorCloningPolicy()
        self.td3_policy = td3_policy or TD3InspiredPolicy()
        self.previous_action: ActionDict | None = None
        self.last_info: dict[str, object] = {}

    def reset(self) -> None:
        self.previous_action = None

    def latent_state(self, obs: ObservationDict) -> np.ndarray:
        return np.asarray(normalized_obs_vec(obs), dtype=float)

    def predict_next_latent(self, obs: ObservationDict, previous_action: ActionDict | None = None) -> np.ndarray:
        prev = self.previous_action if previous_action is None else previous_action
        z = self.latent_state(obs)
        m = predictor_input(obs, prev, bc_policy=self.bc_policy, td3_policy=self.td3_policy)
        return np.clip(z + self.predictor_w @ m, -2.0, 2.0)

    def feature_vector(self, obs: ObservationDict, previous_action: ActionDict | None = None) -> np.ndarray:
        prev = self.previous_action if previous_action is None else previous_action
        z = self.latent_state(obs)
        u = action_to_vec(prev)
        r = setpoint_vec(obs)
        g = td3bc_goal_vec(obs, bc_policy=self.bc_policy, td3_policy=self.td3_policy)
        e = goal_error_vec(obs)
        p = phase_vec(obs)
        z_next_hat = self.predict_next_latent(obs, prev)
        return np.concatenate([z, u, r, g, e, p, z_next_hat, np.ones(1, dtype=float)])

    def probs(self, obs: ObservationDict) -> np.ndarray:
        x = self.feature_vector(obs)
        logits = self.actor_w @ x
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.maximum(exp.sum(), 1e-12)

    def value(self, obs: ObservationDict) -> float:
        return float(self.value_w @ self.feature_vector(obs))

    def action_name(self, obs: ObservationDict) -> str:
        return ACTION_NAMES[int(np.argmax(self.probs(obs)))]

    def remember_action(self, action: ActionDict) -> None:
        self.previous_action = dict(action)

    def act(self, observation: ObservationDict) -> ActionDict:
        probs = self.probs(observation)
        action_idx = int(np.argmax(probs))
        action_name = ACTION_NAMES[action_idx]
        action = safe_discrete_action(action_name, observation)
        td3bc_goal = td3bc_goal_action(observation, bc_policy=self.bc_policy, td3_policy=self.td3_policy)
        self.remember_action(action)
        self.last_info = {
            "selected_strategy": "goal_conditioned_jepa_ppo",
            "policy_action_name": action_name,
            "jepa_goal_error_norm": float(np.linalg.norm(goal_error_vec(observation))),
            "latent_prediction_norm": float(np.linalg.norm(self.predict_next_latent(observation))),
            "operation_setpoint_embedding": setpoint_vec(observation).round(4).tolist(),
            "td3bc_goal_action": dict(td3bc_goal),
            "td3bc_goal_embedding": td3bc_goal_vec(observation, bc_policy=self.bc_policy, td3_policy=self.td3_policy).round(4).tolist(),
            "goal_source": "TD3 smooth target regularized by behavior-cloning expert prior",
            "pipeline": "state_embedding+previous_action_embedding+setpoint_embedding+td3bc_goal_embedding+phase_embedding->jepa_predictor->ppo_policy",
        }
        return action

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez(f, actor_w=self.actor_w, value_w=self.value_w, predictor_w=self.predictor_w)

    @classmethod
    def load(cls, path: Path, bc_path: Path | None = None) -> "GoalConditionedJEPAPPOPolicy":
        load_path = path if path.exists() else Path(f"{path}.npz")
        ckpt = np.load(load_path)
        bc_policy = BehaviorCloningPolicy.load(bc_path) if bc_path is not None and bc_path.exists() else None
        return cls(actor_w=ckpt["actor_w"], value_w=ckpt["value_w"], predictor_w=ckpt["predictor_w"], bc_policy=bc_policy)
