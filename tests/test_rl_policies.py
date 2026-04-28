from __future__ import annotations

from pathlib import Path

from eaf_twin.config.defaults import default_config

from agents.controller import EAFController
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.dqn_policy import DQNPolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.q_learning_policy import QLearningPolicy
from agents.policies.safe_ppo_agentic_mpc import SafePPOAgenticMPCPolicy


REQUIRED = {"power_mw", "oxygen_nm3_min", "ng_nm3_min", "carbon_kg_min", "flux_kg_min", "tap_command"}


def test_rl_policy_actions_valid():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    for pol in [QLearningPolicy(), DQNPolicy(), PPOPolicy(), BehaviorCloningPolicy(), SafePPOAgenticMPCPolicy()]:
        act = pol.act(obs)
        assert REQUIRED.issubset(act.keys())


def test_safe_ppo_has_explainability_fields():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    pol = SafePPOAgenticMPCPolicy()
    pol.act(obs)
    assert "selected_strategy" in pol.last_info
    assert "ppo_raw_action" in pol.last_info
