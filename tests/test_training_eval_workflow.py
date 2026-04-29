from __future__ import annotations

from pathlib import Path

import numpy as np


def test_docker_compose_commands_cover_all_training_and_eval():
    compose = Path("docker-compose.yml").read_text()
    assert "--algorithm\", \"all\"" in compose
    assert "results/agent_training" in compose
    assert "--training-dir\", \"results/agent_training\"" in compose


def test_run_agent_requires_missing_rl_checkpoints_by_default(tmp_path):
    from agents.runners.run_agent import _build_policies

    class Args:
        training_dir = tmp_path
        include_rl_baselines = True
        mpc_horizon = 8
        allow_missing_rl_baselines = True

    policies, missing, status = _build_policies(Args())
    assert "ppo" in missing
    assert status["ppo"] == "missing checkpoint"
    assert "ppo" not in policies


def test_ppo_nonzero_checkpoint_roundtrip(tmp_path):
    from agents.policies.ppo_policy import PPOPolicy

    p = PPOPolicy()
    p.actor_w[:] = 1.0
    path = tmp_path / "best_policy.pt"
    p.save(path)
    data = np.load(path, allow_pickle=False)
    assert np.any(np.abs(data["actor_w"]) > 0.0)
