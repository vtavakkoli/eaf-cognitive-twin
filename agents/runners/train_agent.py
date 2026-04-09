from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eaf_twin.config.defaults import scenario_configs
from eaf_twin.config.loader import load_config

from agents.controller import EAFController
from agents.policies.heuristic import HeuristicParams
from agents.policies.trainable_policy import TrainablePolicy
from agents.runners.episode_runner import run_episode


def evaluate_params(base_cfg, params: HeuristicParams, scenario_names: list[str]) -> tuple[float, list[dict]]:
    scenarios = scenario_configs(base_cfg)
    policy = TrainablePolicy(params=params)
    details = []
    total = 0.0
    for name in scenario_names:
        controller = EAFController(replace(scenarios[name]), enhanced_model=True)
        out = run_episode(controller, policy, policy_name="train_eval")
        total += out.total_reward
        details.append({"scenario": name, "reward": out.total_reward, "steps": out.steps, "tapped_kg": out.final_tapped_kg})
    return total / max(len(scenario_names), 1), details


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/tune an external EAF control agent")
    parser.add_argument("--config", type=Path, default=Path("configs/base_case.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/agent_training"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.config)
    scenario_names = ["base_case", "higher_oxygen", "higher_natural_gas", "improved_foamy_slag", "delayed_melting_downtime"]

    rng = np.random.default_rng(args.seed)
    best_score = float("-inf")
    best_params = HeuristicParams()
    logs = []

    for i in range(1, args.iterations + 1):
        candidate = HeuristicParams(
            early_power_mw=float(rng.uniform(80, 120)),
            mid_power_mw=float(rng.uniform(55, 95)),
            late_power_mw=float(rng.uniform(15, 45)),
            oxygen_scale=float(rng.uniform(0.8, 1.25)),
            gas_scale=float(rng.uniform(0.75, 1.3)),
        )
        score, details = evaluate_params(base_cfg, candidate, scenario_names)
        is_best = score > best_score
        if is_best:
            best_score = score
            best_params = candidate
            TrainablePolicy(params=best_params).save(out / "checkpoints" / "best_policy.json")

        logs.append({"iteration": i, "mean_reward": score, "is_best": is_best, **candidate.__dict__})
        (out / "checkpoints" / f"iter_{i:03d}.json").write_text(json.dumps({"score": score, "params": candidate.__dict__, "details": details}, indent=2))

    log_df = pd.DataFrame(logs)
    log_df.to_csv(out / "training_log.csv", index=False)
    summary = {
        "iterations": args.iterations,
        "best_score": best_score,
        "best_params": best_params.__dict__,
        "scenarios": scenario_names,
        "config": str(args.config),
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(8, 4))
    plt.plot(log_df["iteration"], log_df["mean_reward"], marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Mean reward")
    plt.title("Agent training progress")
    plt.tight_layout()
    plt.savefig(out / "training_reward_curve.png", dpi=150)


if __name__ == "__main__":
    main()
