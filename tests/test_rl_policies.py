from __future__ import annotations

from pathlib import Path

from eaf_twin.config.defaults import default_config

from agents.controller import EAFController
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.dqn_policy import DQNPolicy
from agents.policies.goal_conditioned_jepa_ppo import GoalConditionedJEPAPPOPolicy, FEATURE_DIM, TD3BC_GOAL_VECTOR_DIM, td3bc_goal_vec
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.q_learning_policy import QLearningPolicy
from agents.policies.safe_ppo_agentic_mpc import SafePPOAgenticMPCPolicy
from agents.policies.safe_ppo_agentic_td3_bc import SafePPOAgenticTD3BCPolicy


REQUIRED = {"power_mw", "oxygen_nm3_min", "ng_nm3_min", "carbon_kg_min", "flux_kg_min", "tap_command"}


def test_rl_policy_actions_valid():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    for pol in [QLearningPolicy(), DQNPolicy(), PPOPolicy(), GoalConditionedJEPAPPOPolicy(), BehaviorCloningPolicy(), SafePPOAgenticMPCPolicy(), SafePPOAgenticTD3BCPolicy()]:
        act = pol.act(obs)
        assert REQUIRED.issubset(act.keys())


def test_safe_ppo_has_explainability_fields():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    pol = SafePPOAgenticMPCPolicy()
    pol.act(obs)
    assert "selected_strategy" in pol.last_info
    assert "ppo_raw_action" in pol.last_info


def test_goal_conditioned_jepa_ppo_feature_shape():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    pol = GoalConditionedJEPAPPOPolicy()
    assert pol.feature_vector(obs).shape[0] == FEATURE_DIM
    assert pol.actor_w.shape[1] == FEATURE_DIM


def test_goal_conditioned_jepa_ppo_uses_td3bc_goal_setting():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    pol = GoalConditionedJEPAPPOPolicy()
    act = pol.act(obs)
    assert REQUIRED.issubset(act.keys())
    assert td3bc_goal_vec(obs).shape[0] == TD3BC_GOAL_VECTOR_DIM
    assert "td3bc_goal_action" in pol.last_info
    assert "td3bc_goal_embedding" in pol.last_info
    assert "ppo_safeagent_td3bc_backbone_action" in pol.last_info
    assert "jepa_residual_reason" in pol.last_info
    assert "jepa_predictor" in pol.last_info["pipeline"]


def test_jepa_residual_keeps_tap_ready_backbone_and_blocks_invalid_tap():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    obs.update({
        "time_min": 58.0,
        "bath_temp_c": 1590.0,
        "melted_fraction": 0.94,
        "liquid_steel_kg": 94000.0,
        "can_tap": False,
    })
    pol = GoalConditionedJEPAPPOPolicy()
    act = pol.act(obs)
    assert REQUIRED.issubset(act.keys())
    assert act["tap_command"] is False
    assert act["power_mw"] >= 70.0
    assert pol.last_info["jepa_residual_reason"] in {"jepa_late_tap_ready_recovery", "jepa_mass_protection"}
