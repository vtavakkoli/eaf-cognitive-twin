from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from agents.base import BasePolicy
from agents.controller import EAFController


@dataclass
class EpisodeOutcome:
    episode_df: pd.DataFrame
    total_reward: float
    steps: int
    final_tapped_kg: float


def run_episode(
    controller: EAFController,
    policy: BasePolicy,
    policy_name: str,
    max_steps: int | None = None,
    max_minutes: float = 70.0,
) -> EpisodeOutcome:
    if max_steps is None:
        dt_s = float(getattr(controller.config, "dt_s", 1.0))
        max_steps = max(1, int((max_minutes * 60.0) / max(dt_s, 1e-9)))
    obs = controller.reset()
    policy.reset()
    rows = []
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = policy.act(obs)
        result = controller.step(action)
        row_obs = {k: v for k, v in result.observation.items() if not k.startswith("_") and k != "default_schedule_action"}
        row = {"policy": policy_name, **row_obs, **result.info["safe_action"], "reward": result.reward}
        row.update({k: v for k, v in result.info.items() if k in {"safety_violation", "temperature_violation", "carbon_violation", "invalid_tap_command", "action_clamped", "clamp_reason", "is_downtime"}})
        if hasattr(policy, "tap_reason"):
            row["tap_reason"] = getattr(policy, "tap_reason")(obs)
        rows.append(row)
        obs = result.observation
        total_reward += result.reward
        done = result.done
        steps += 1

    df = pd.DataFrame(rows)
    if not df.empty and steps >= max_steps and not done:
        df.loc[df.index[-1], "termination_reason"] = "max_steps_reached"
    final_tapped = float(df["cum_tapped_kg"].iloc[-1]) if not df.empty else 0.0
    return EpisodeOutcome(episode_df=df, total_reward=total_reward, steps=steps, final_tapped_kg=final_tapped)
