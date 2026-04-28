from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eaf_twin.config.loader import load_config

from agents.policies.baseline_schedule import IndustrialBaselineSchedulePolicy
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.dqn_policy import DQNPolicy
from agents.policies.mpc_policy import MPCPolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.q_learning_policy import QLearningPolicy
from agents.policies.rule_based import RuleBasedPolicy
from agents.policies.safe_ppo_agentic_mpc import SafePPOAgenticMPCPolicy
from agents.policies.trainable_policy import TrainablePolicy
from agents.runners.benchmark_runner import run_benchmark


def _safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return pd.Series(["n/a" if abs(float(d)) < 1e-9 else 100.0 * float(n) / float(d) for n, d in zip(num, den)])


def _ci95(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


def _normalized_score(df: pd.DataFrame) -> pd.Series:
    out = (
        0.45 * (df["total_reward"] - df["total_reward"].min()) / max(df["total_reward"].max() - df["total_reward"].min(), 1e-9)
        + 0.25 * df["tap_success"].astype(float)
        + 0.15 * (1.0 - (df["energy_per_ton"] - df["energy_per_ton"].min()) / max(df["energy_per_ton"].max() - df["energy_per_ton"].min(), 1e-9))
        + 0.10 * (1.0 - (df["constraint_violation_rate"] - df["constraint_violation_rate"].min()) / max(df["constraint_violation_rate"].max() - df["constraint_violation_rate"].min(), 1e-9))
        + 0.05 * (1.0 - (df["tap_temperature_error"] - df["tap_temperature_error"].min()) / max(df["tap_temperature_error"].max() - df["tap_temperature_error"].min(), 1e-9))
    )
    return out


def _policy_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    agg = summary_df.groupby("policy", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        std_reward=("total_reward", "std"),
        median_reward=("total_reward", "median"),
        reward_ci95=("total_reward", _ci95),
        success_rate=("tap_success", "mean"),
        success_std=("tap_success", "std"),
        mean_tapped_kg=("cum_tapped_kg", "mean"),
        mean_electric_mwh=("cum_electric_mwh", "mean"),
        mean_oxygen_nm3=("cum_oxygen_nm3", "mean"),
        mean_ng_nm3=("cum_ng_nm3", "mean"),
        energy_per_ton=("energy_per_ton", "mean"),
        oxygen_per_ton=("oxygen_per_ton", "mean"),
        natural_gas_per_ton=("natural_gas_per_ton", "mean"),
        tap_temperature_error=("tap_temperature_error", "mean"),
        mass_error=("mass_error", "mean"),
        carbon_error=("carbon_error", "mean"),
        constraint_violation_rate=("constraint_violation_rate", "mean"),
        invalid_action_rate=("invalid_action_rate", "mean"),
        temperature_violation_count=("temperature_violation_count", "sum"),
        safety_violation_count=("safety_violation_count", "sum"),
        invalid_tap_count=("invalid_tap_count", "sum"),
        action_clamp_count=("action_clamp_count", "sum"),
    )
    agg["normalized_score"] = _normalized_score(agg.rename(columns={"mean_reward": "total_reward", "success_rate": "tap_success"}))
    return agg.sort_values("normalized_score", ascending=False)


def _paired_stats(df: pd.DataFrame, baseline_policy: str, target_policy: str) -> dict[str, object] | None:
    b = df[df.policy == baseline_policy][["seed", "scenario", "total_reward"]].rename(columns={"total_reward": "baseline"})
    t = df[df.policy == target_policy][["seed", "scenario", "total_reward"]].rename(columns={"total_reward": "target"})
    m = b.merge(t, on=["seed", "scenario"])
    if m.empty:
        return None
    diff = m["target"] - m["baseline"]
    effect = float(diff.mean() / (diff.std(ddof=1) + 1e-9))
    pval = np.nan
    ci_low, ci_high = np.nan, np.nan
    try:
        from scipy import stats

        pval = float(stats.ttest_rel(m["target"], m["baseline"]).pvalue)
    except Exception:
        boots = []
        arr = diff.to_numpy()
        rng = np.random.default_rng(7)
        for _ in range(500):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boots.append(np.mean(sample))
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return {
        "baseline": baseline_policy,
        "target": target_policy,
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "p_value": pval,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "cohens_d": effect,
    }


def _plot_all_figures(summary_df: pd.DataFrame, output_dir: Path) -> list[str]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    policy = summary_df.groupby("policy", as_index=False).agg(mean_reward=("total_reward", "mean"), std_reward=("total_reward", "std"), tap_success_rate=("tap_success", "mean"), energy_per_ton=("energy_per_ton", "mean"), violation_count=("temperature_violation_count", "sum"), normalized_score=("normalized_score", "mean"))

    for col, fn, title in [
        ("mean_reward", "reward_mean_std.png", "Reward mean by policy"),
        ("tap_success_rate", "tap_success_rate.png", "Tap success rate by policy"),
        ("energy_per_ton", "energy_per_ton.png", "Energy per ton by policy"),
        ("violation_count", "violation_count.png", "Violation count by policy"),
        ("normalized_score", "normalized_score_ranking.png", "Normalized score by policy"),
    ]:
        ax = policy.sort_values(col, ascending=False).plot(x="policy", y=col, kind="bar", legend=False, figsize=(9, 4))
        ax.set_title(title)
        plt.tight_layout(); plt.savefig(fig_dir / fn, dpi=150); plt.close(); files.append(fn)

    pivot = summary_df.pivot_table(index="scenario", columns="policy", values="normalized_score", aggfunc="mean")
    plt.figure(figsize=(10, 4)); plt.imshow(pivot.values, aspect="auto", cmap="magma"); plt.colorbar(label="normalized_score")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right"); plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Scenario-policy heatmap"); plt.tight_layout(); plt.savefig(fig_dir / "scenario_policy_heatmap.png", dpi=150); plt.close(); files.append("scenario_policy_heatmap.png")

    plt.figure(figsize=(8, 4));
    for p, g in summary_df.groupby("policy"):
        plt.scatter(g["energy_per_ton"].mean(), g["total_reward"].mean(), label=p)
    plt.legend(); plt.xlabel("energy_per_ton"); plt.ylabel("total_reward"); plt.title("Pareto: reward vs energy_per_ton")
    plt.tight_layout(); plt.savefig(fig_dir / "pareto_reward_vs_energy_per_ton.png", dpi=150); plt.close(); files.append("pareto_reward_vs_energy_per_ton.png")

    # temperature trajectory comparison on base_case seed0 if available
    ts_dir = output_dir / "timeseries"
    plt.figure(figsize=(9, 4))
    for path in sorted(ts_dir.glob("agent_timeseries_base_case_*_seed0.csv")):
        p = path.stem.split("_seed")[0].split("agent_timeseries_base_case_")[1]
        d = pd.read_csv(path)
        if {"time_min", "bath_temp_c"}.issubset(d.columns):
            plt.plot(d["time_min"], d["bath_temp_c"], label=p)
    plt.legend(fontsize=7); plt.title("Temperature trajectory comparison (base_case, seed0)"); plt.xlabel("time_min"); plt.ylabel("bath_temp_c")
    plt.tight_layout(); plt.savefig(fig_dir / "temperature_trajectory_comparison.png", dpi=150); plt.close(); files.append("temperature_trajectory_comparison.png")
    return files


def _render_html(output_dir: Path, summary_df: pd.DataFrame, policy_stats: pd.DataFrame, scenario_rank: pd.DataFrame, comparison_df: pd.DataFrame, stat_tests: pd.DataFrame, figures: list[str], max_steps: int, dt_s: float) -> None:
    best = policy_stats.iloc[0]
    policy_list = ", ".join(sorted(summary_df["policy"].unique()))
    html = f"""
<html><head><meta charset='utf-8'><title>EAF Agentic AI Benchmark: PPO, Q-Learning, DQN, MPC, and Proposed Safe PPO-Agentic MPC</title></head>
<body style='font-family:Arial;margin:20px'>
<h1>EAF Agentic AI Benchmark: PPO, Q-Learning, DQN, MPC, and Proposed Safe PPO-Agentic MPC</h1>
<p>All policies are evaluated on the same enhanced hybrid first-principles simulator, Model C.</p>
<p>Simulation budget: {max_steps} steps, dt_s = {dt_s} seconds, equivalent to {max_steps*dt_s/60:.1f} simulated minutes.</p>
<p>Policies: {policy_list}</p>
<h2>Main result table (mean ± std)</h2>{policy_stats.to_html(index=False)}
<h2>Scenario-level ranking table</h2>{scenario_rank.to_html(index=False)}
<h2>Baseline comparison table</h2>{comparison_df.to_html(index=False)}
<h2>Statistical significance table</h2>{stat_tests.to_html(index=False)}
<h2>Best policy decision</h2><p>Best policy by normalized_score: <b>{best['policy']}</b> (score={best['normalized_score']:.4f}), selected by transparent weighted metric combining reward, success, efficiency, violations, and temperature error.</p>
<h2>Figures</h2>{''.join([f"<div><img src='figures/{f}' style='max-width:1000px;width:100%'></div>" for f in figures])}
</body></html>
"""
    (output_dir / "result.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained/default agent benchmark over EAF scenarios")
    parser.add_argument("--config", type=Path, default=Path("configs/base_case.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/agent_run"))
    parser.add_argument("--trained-policy", type=Path, default=Path("results/agent_training/checkpoints/best_policy.json"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-scenarios", type=int, default=6)
    parser.add_argument("--model", choices=["C"], default="C")
    parser.add_argument("--mpc-horizon", type=int, default=8)
    parser.add_argument("--report-format", default="html,csv,md")
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--include-rl-baselines", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir; output_dir.mkdir(parents=True, exist_ok=True)

    policies = {"baseline_schedule": IndustrialBaselineSchedulePolicy(), "rule_based": RuleBasedPolicy(), "mpc": MPCPolicy(horizon=args.mpc_horizon)}
    if args.trained_policy.exists():
        policies["agentic_ai"] = TrainablePolicy.load(args.trained_policy)
    if args.include_rl_baselines:
        if Path("results/agent_training/q_learning/q_table.json").exists(): policies["q_learning"] = QLearningPolicy.load(Path("results/agent_training/q_learning/q_table.json"))
        else: policies["q_learning"] = QLearningPolicy()
        if Path("results/agent_training/dqn/best_policy.npy").exists(): policies["dqn"] = DQNPolicy.load(Path("results/agent_training/dqn/best_policy.npy"))
        else: policies["dqn"] = DQNPolicy()
        if Path("results/agent_training/ppo/best_policy.pt").exists(): policies["ppo"] = PPOPolicy.load(Path("results/agent_training/ppo/best_policy.pt"))
        else: policies["ppo"] = PPOPolicy()
        if Path("results/agent_training/behavior_cloning/policy.json").exists(): policies["behavior_cloning"] = BehaviorCloningPolicy.load(Path("results/agent_training/behavior_cloning/policy.json"))
        else: policies["behavior_cloning"] = BehaviorCloningPolicy()
        if Path("results/agent_training/safe_ppo_agentic_mpc/best_policy.pt").exists():
            policies["safe_ppo_agentic_mpc"] = SafePPOAgenticMPCPolicy.load(Path("results/agent_training/safe_ppo_agentic_mpc/best_policy.pt"), horizon=args.mpc_horizon)
        else:
            policies["safe_ppo_agentic_mpc"] = SafePPOAgenticMPCPolicy(horizon=args.mpc_horizon)

    seeds = list(range(args.seeds))
    scenario_order = ["base_case", "higher_oxygen", "higher_natural_gas", "improved_foamy_slag", "dri20", "delayed_melting_downtime"][: args.n_scenarios]
    summary_df = run_benchmark(args.config, policies, output_dir, seeds=seeds, selected_scenarios=scenario_order, max_steps=args.max_steps)
    summary_df["normalized_score"] = _normalized_score(summary_df)
    summary_df = summary_df.sort_values(["scenario", "total_reward"], ascending=[True, False]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    policy_stats = _policy_stats(summary_df)
    policy_stats["overall_rank"] = np.arange(1, len(policy_stats) + 1)
    policy_stats.to_csv(output_dir / "policy_aggregate_summary.csv", index=False)

    summary_df["rank_by_scenario"] = summary_df.groupby(["seed", "scenario"])["normalized_score"].rank(ascending=False, method="dense")
    scenario_rank = summary_df.groupby(["scenario", "policy"], as_index=False)["rank_by_scenario"].mean().sort_values(["scenario", "rank_by_scenario"])

    kpi_comparison = summary_df.pivot_table(index=["seed", "scenario"], columns="policy", values=["cum_tapped_kg", "total_reward", "cum_electric_mwh", "cum_oxygen_nm3", "cum_ng_nm3", "final_temp_c", "normalized_score"]) 
    kpi_comparison.to_csv(output_dir / "kpi_comparison.csv")

    baseline = summary_df[summary_df["policy"] == "baseline_schedule"][["seed", "scenario", "total_reward", "cum_tapped_kg", "cum_electric_mwh"]]
    comparison_rows = []
    for policy in sorted(summary_df["policy"].unique()):
        if policy == "baseline_schedule":
            continue
        cur = summary_df[summary_df["policy"] == policy][["seed", "scenario", "total_reward", "cum_tapped_kg", "cum_electric_mwh"]]
        j = cur.merge(baseline, on=["seed", "scenario"], suffixes=("_policy", "_baseline"))
        if j.empty:
            continue
        reward_gain = _safe_pct(j["total_reward_policy"] - j["total_reward_baseline"], j["total_reward_baseline"].abs())
        comparison_rows.append({"policy": policy, "reward_delta": float((j["total_reward_policy"] - j["total_reward_baseline"]).mean()), "tapped_delta_kg": float((j["cum_tapped_kg_policy"] - j["cum_tapped_kg_baseline"]).mean()), "electric_delta_mwh": float((j["cum_electric_mwh_policy"] - j["cum_electric_mwh_baseline"]).mean()), "reward_gain_vs_baseline_pct": "n/a" if (reward_gain == "n/a").any() else float(pd.to_numeric(reward_gain).mean())})
    comparison_df = pd.DataFrame(comparison_rows)

    stat_df = summary_df.groupby("policy", as_index=False).agg(mean=("total_reward", "mean"), std=("total_reward", "std"), median=("total_reward", "median"), min=("total_reward", "min"), max=("total_reward", "max"))
    stat_df.to_csv(output_dir / "statistical_analysis.csv", index=False)

    pairs = []
    for target in sorted(summary_df["policy"].unique()):
        for baseline_name in ["baseline_schedule", "mpc", "ppo"]:
            if target == baseline_name or baseline_name not in summary_df["policy"].unique():
                continue
            r = _paired_stats(summary_df, baseline_name, target)
            if r:
                pairs.append(r)
    stat_tests = pd.DataFrame(pairs)
    stat_tests.to_csv(output_dir / "statistical_tests.csv", index=False)

    figures = _plot_all_figures(summary_df, output_dir)
    _render_html(output_dir, summary_df, policy_stats, scenario_rank, comparison_df, stat_tests, figures, args.max_steps, float(load_config(args.config).dt_s))

    (output_dir / "report.md").write_text("# Agent Run Report\n\nSee result.html and CSV artifacts for full benchmark.")
    dt_s = float(load_config(args.config).dt_s)
    (output_dir / "run_manifest.json").write_text(json.dumps({"policies": list(policies.keys()), "config": str(args.config), "model_name": "Model_C_enhanced_hybrid", "max_steps": args.max_steps, "dt_s": dt_s, "simulated_minutes": args.max_steps * dt_s / 60.0, "early_termination_allowed": True}, indent=2))


if __name__ == "__main__":
    main()
