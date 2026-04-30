from __future__ import annotations

import torch

from eaf_twin.config.defaults import default_config

from agents.controller import EAFController
from agents.policies.dqn_policy import DQNPolicy
from agents.policies.safe_ppo_agentic_mpc import SafePPOAgenticMPCPolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.rl_common import ACTION_NAMES, safe_discrete_action
from agents.runners.train_agent import _action_index_from_action


def test_reward_uses_incremental_consumption():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    act = safe_discrete_action(ACTION_NAMES[0], obs)
    r1 = ctrl.step(act).reward
    r2 = ctrl.step(act).reward
    assert abs(r2) < abs(r1) * 3.0


def test_safe_ppo_uses_executed_action_index():
    ctrl = EAFController(default_config(), enhanced_model=True)
    obs = ctrl.reset()
    pol = SafePPOAgenticMPCPolicy(PPOPolicy())
    executed = pol.act(obs)
    idx = _action_index_from_action(obs, executed)
    assert 0 <= idx < len(ACTION_NAMES)


def test_dqn_is_torch_module():
    pol = DQNPolicy()
    assert isinstance(pol.q_network, torch.nn.Module)
    assert sum(p.numel() for p in pol.q_network.parameters() if p.requires_grad) > 0


def test_dqn_target_network_update():
    online = DQNPolicy()
    target = DQNPolicy()
    target.q_network.load_state_dict(online.q_network.state_dict())
    with torch.no_grad():
        for p in online.q_network.parameters():
            p.add_(0.1)
    target.q_network.load_state_dict(online.q_network.state_dict())
    for p1, p2 in zip(online.q_network.parameters(), target.q_network.parameters()):
        assert torch.allclose(p1, p2)
