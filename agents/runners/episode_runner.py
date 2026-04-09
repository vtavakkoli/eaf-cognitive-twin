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


def run_episode(controller: EAFController, policy: BasePolicy, policy_name: str) -> EpisodeOutcome:
    obs = controller.reset()
    policy.reset()
    rows = []
    total_reward = 0.0
    done = False
    while not done:
        action = policy.act(obs)
        result = controller.step(action)
        row = {"policy": policy_name, **result.observation, **result.info["safe_action"], "reward": result.reward}
        rows.append(row)
        obs = result.observation
        total_reward += result.reward
        done = result.done
    df = pd.DataFrame(rows)
    final_tapped = float(df["cum_tapped_kg"].iloc[-1]) if not df.empty else 0.0
    return EpisodeOutcome(episode_df=df, total_reward=total_reward, steps=len(rows), final_tapped_kg=final_tapped)
