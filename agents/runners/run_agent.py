from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eaf_twin.config.loader import load_config

from agents.base import BasePolicy
from agents.policies.baseline_schedule import IndustrialBaselineSchedulePolicy
from agents.policies.behavior_cloning_policy import BehaviorCloningPolicy
from agents.policies.dqn_policy import DQNPolicy
from agents.policies.mpc_policy import MPCPolicy
from agents.policies.ppo_policy import PPOPolicy
from agents.policies.q_learning_policy import QLearningPolicy
from agents.policies.rule_based import RuleBasedPolicy
from agents.policies.sac_inspired_policy import SACInspiredPolicy
from agents.policies.safe_ppo_agentic_mpc import SafePPOAgenticMPCPolicy
from agents.policies.safe_ppo_agentic_sac import SafePPOAgenticSACPolicy
from agents.policies.safe_ppo_agentic_td3 import SafePPOAgenticTD3Policy
from agents.policies.td3_inspired_policy import TD3InspiredPolicy
from agents.policies.trainable_policy import TrainablePolicy
from agents.runners.benchmark_runner import run_benchmark

POLICY_LABELS = {
    "baseline_schedule": "Baseline Schedule",
    "rule_based": "Rule-Based",
    "mpc": "MPC",
    "trainable_adaptive_controller": "Trainable Adaptive Controller",
    "q_learning": "Q-Learning",
    "dqn": "DQN",
    "ppo": "PPO",
    "safe_ppo_agentic_mpc": "Proposed Safe PPO-Agentic MPC",
    "safe_ppo_agentic_sac": "Proposed Safe PPO-Agentic SAC",
    "safe_ppo_agentic_td3": "Proposed Safe PPO-Agentic TD3",
    "behavior_cloning": "Behavior Cloning",
    "sac_inspired": "SAC-Inspired",
    "td3_inspired": "TD3-Inspired",
}


def _canonical_policy_key(policy_key: str) -> str:
    if policy_key == "agentic_ai":
        return "trainable_adaptive_controller"
    return policy_key


def _display_name(policy_key: str) -> str:
    canonical = _canonical_policy_key(policy_key)
    return POLICY_LABELS.get(canonical, canonical.replace("_", " ").title())


def _safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return pd.Series(["n/a" if abs(float(d)) < 1e-9 else 100.0 * float(n) / float(d) for n, d in zip(num, den)])


def _ci95(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


def _normalized_score(df: pd.DataFrame) -> pd.Series:
    energy = pd.to_numeric(df["energy_per_ton"], errors="coerce")
    energy_filled = energy.fillna(energy.max(skipna=True) if energy.notna().any() else 0.0)
    target_tapped_kg = pd.to_numeric(df["target_tapped_kg"], errors="coerce").replace(0, np.nan)
    production_norm = (pd.to_numeric(df["tapped_kg"], errors="coerce") / target_tapped_kg).clip(lower=0.0, upper=1.0).fillna(0.0)
    production_eff = (pd.to_numeric(df["tapped_kg"], errors="coerce") / energy_filled.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    production_eff_filled = production_eff.fillna(production_eff.min(skipna=True) if production_eff.notna().any() else 0.0)
    production_efficiency_norm = (production_eff_filled - production_eff_filled.min()) / max(production_eff_filled.max() - production_eff_filled.min(), 1e-9)
    energy_per_ton_norm = 1.0 - (energy_filled - energy_filled.min()) / max(energy_filled.max() - energy_filled.min(), 1e-9)
    temp_norm = 1.0 - (df["tap_temperature_error"] - df["tap_temperature_error"].min()) / max(df["tap_temperature_error"].max() - df["tap_temperature_error"].min(), 1e-9)
    carbon_norm = 1.0 - (df["carbon_error"] - df["carbon_error"].min()) / max(df["carbon_error"].max() - df["carbon_error"].min(), 1e-9)
    quality_norm = 0.5 * temp_norm + 0.5 * carbon_norm
    safety_norm = 1.0 - (df["constraint_violation_rate"] - df["constraint_violation_rate"].min()) / max(df["constraint_violation_rate"].max() - df["constraint_violation_rate"].min(), 1e-9)
    tap_ready_norm = df["tap_ready"].astype(float)
    return (
        0.25 * tap_ready_norm
        + 0.20 * production_norm
        + 0.20 * production_efficiency_norm
        + 0.15 * energy_per_ton_norm
        + 0.10 * quality_norm
        + 0.10 * safety_norm
    )


def _policy_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    agg = summary_df.groupby("policy", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        std_reward=("total_reward", "std"),
        median_reward=("total_reward", "median"),
        reward_ci95=("total_reward", _ci95),
        tap_ready_rate=("tap_ready", "mean"),
        tap_ready_std=("tap_ready", "std"),
        mean_tapped_kg=("tapped_kg", "mean"),
        mean_target_tapped_kg=("target_tapped_kg", "mean"),
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
    agg["tapped_kg"] = agg["mean_tapped_kg"]
    agg["target_tapped_kg"] = agg["mean_target_tapped_kg"]
    agg["tap_ready"] = agg["tap_ready_rate"] >= 0.5
    agg["normalized_score"] = _normalized_score(agg)
    agg["production_norm"] = (agg["mean_tapped_kg"] / agg["mean_target_tapped_kg"].replace(0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0)
    agg["tap_ready_norm"] = agg["tap_ready_rate"].astype(float)
    production_eff = (agg["mean_tapped_kg"] / agg["energy_per_ton"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    agg["production_efficiency_norm"] = (production_eff - production_eff.min()) / max(production_eff.max() - production_eff.min(), 1e-9)
    agg["energy_per_ton_norm"] = 1.0 - (agg["energy_per_ton"] - agg["energy_per_ton"].min()) / max(agg["energy_per_ton"].max() - agg["energy_per_ton"].min(), 1e-9)
    agg["safety_norm"] = 1.0 - (agg["constraint_violation_rate"] - agg["constraint_violation_rate"].min()) / max(agg["constraint_violation_rate"].max() - agg["constraint_violation_rate"].min(), 1e-9)
    temp_norm = 1.0 - (agg["tap_temperature_error"] - agg["tap_temperature_error"].min()) / max(agg["tap_temperature_error"].max() - agg["tap_temperature_error"].min(), 1e-9)
    carbon_norm = 1.0 - (agg["carbon_error"] - agg["carbon_error"].min()) / max(agg["carbon_error"].max() - agg["carbon_error"].min(), 1e-9)
    agg["quality_norm"] = 0.5 * temp_norm + 0.5 * carbon_norm
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


def _format_table(df: pd.DataFrame) -> str:
    pretty = df.copy()
    if "policy" in pretty.columns:
        pretty["policy"] = pretty["policy"].map(_display_name)
    return pretty.to_html(index=False, classes="styled-table", na_rep="N/A")


def _build_policy_coverage(summary_df: pd.DataFrame, evaluated_policies: list[str]) -> pd.DataFrame:
    grouped = summary_df.groupby("policy", as_index=False).agg(
        episodes=("policy", "size"),
        scenarios=("scenario", "nunique"),
        seeds=("seed", "nunique"),
        tap_ready_rate=("tap_ready", "mean"),
    )
    coverage = pd.DataFrame({"policy": evaluated_policies}).merge(grouped, on="policy", how="left")
    for col in ["episodes", "scenarios", "seeds"]:
        coverage[col] = coverage[col].fillna(0).astype(int)
    coverage["tap_ready_rate"] = pd.to_numeric(coverage["tap_ready_rate"], errors="coerce")
    coverage["display_name"] = coverage["policy"].map(_display_name)
    coverage["included_in_all_outputs"] = True
    return coverage.sort_values("policy")


def _plot_all_figures(summary_df: pd.DataFrame, output_dir: Path, evaluated_policies: list[str]) -> list[str]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    policy = summary_df.groupby("policy", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        std_reward=("total_reward", "std"),
        tap_ready_rate=("tap_ready", "mean"),
        energy_per_ton=("energy_per_ton", "mean"),
        violation_count=("temperature_violation_count", "sum"),
        normalized_score=("normalized_score", "mean"),
    )
    policy = policy.set_index("policy").reindex(evaluated_policies).reset_index()
    policy["display_name"] = policy["policy"].map(_display_name)

    for col, fn, title in [
        ("mean_reward", "reward_mean_std.png", "Reward mean by policy"),
        ("tap_ready_rate", "tap_ready_rate.png", "Tap ready rate by policy"),
        ("energy_per_ton", "energy_per_ton.png", "Energy per ton by policy"),
        ("violation_count", "violation_count.png", "Violation count by policy"),
        ("normalized_score", "normalized_score_ranking.png", "Normalized score by policy"),
    ]:
        sorted_policy = policy.sort_values(col, ascending=False, na_position="last")
        ax = sorted_policy.plot(x="display_name", y=col, kind="bar", legend=False, figsize=(11, 4), color="#4f81bd")
        for patch, policy_key in zip(ax.patches, sorted_policy["policy"]):
            if policy_key == "safe_ppo_agentic_mpc":
                patch.set_color("#d62728")
        ax.set_title(title)
        ax.set_xlabel("")
        plt.tight_layout()
        plt.savefig(fig_dir / fn, dpi=150)
        plt.close()
        files.append(fn)

    pivot = summary_df.pivot_table(index="scenario", columns="policy", values="normalized_score", aggfunc="mean").reindex(columns=evaluated_policies)
    plt.figure(figsize=(12, 4))
    plt.imshow(pivot.values, aspect="auto", cmap="magma")
    plt.colorbar(label="normalized_score")
    plt.xticks(range(len(pivot.columns)), [_display_name(p) for p in pivot.columns], rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Scenario-policy heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir / "scenario_policy_heatmap.png", dpi=150)
    plt.close()
    files.append("scenario_policy_heatmap.png")

    plt.figure(figsize=(9, 4))
    for p in evaluated_policies:
        g = summary_df[summary_df["policy"] == p]
        if g.empty:
            continue
        marker = "*" if p == "safe_ppo_agentic_mpc" else "o"
        size = 180 if p == "safe_ppo_agentic_mpc" else 80
        plt.scatter(g["energy_per_ton"].mean(), g["total_reward"].mean(), label=_display_name(p), marker=marker, s=size)
    plt.legend(fontsize=8)
    plt.xlabel("energy_per_ton")
    plt.ylabel("total_reward")
    plt.title("Pareto: reward vs energy_per_ton")
    plt.tight_layout()
    plt.savefig(fig_dir / "pareto_reward_vs_energy_per_ton.png", dpi=150)
    plt.close()
    files.append("pareto_reward_vs_energy_per_ton.png")

    ts_dir = output_dir / "timeseries"
    plt.figure(figsize=(10, 4))
    for p in evaluated_policies:
        path = ts_dir / f"agent_timeseries_base_case_{p}_seed0.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if {"time_min", "bath_temp_c"}.issubset(d.columns):
            linewidth = 2.5 if p == "safe_ppo_agentic_mpc" else 1.2
            plt.plot(d["time_min"], d["bath_temp_c"], label=_display_name(p), linewidth=linewidth)
    plt.legend(fontsize=7)
    plt.title("Temperature trajectory comparison (base_case, seed0)")
    plt.xlabel("time_min")
    plt.ylabel("bath_temp_c")
    plt.tight_layout()
    plt.savefig(fig_dir / "temperature_trajectory_comparison.png", dpi=150)
    plt.close()
    files.append("temperature_trajectory_comparison.png")
    return files


def _render_html(
    output_dir: Path,
    summary_df: pd.DataFrame,
    policy_stats: pd.DataFrame,
    scenario_rank: pd.DataFrame,
    comparison_df: pd.DataFrame,
    stat_tests: pd.DataFrame,
    policy_coverage: pd.DataFrame,
    figures: list[str],
    max_steps: int,
    dt_s: float,
    warnings: list[str],
) -> None:
    best = policy_stats.iloc[0]
    policy_list = ", ".join(_display_name(p) for p in sorted(summary_df["policy"].unique()))
    warning_html = "".join([f"<li>{w}</li>" for w in warnings]) if warnings else "<li>No warnings detected.</li>"
    kpis = {
        "Policies Evaluated": int(summary_df["policy"].nunique()),
        "Scenarios": int(summary_df["scenario"].nunique()),
        "Seeds": int(summary_df["seed"].nunique()),
        "Tap Ready (overall)": f"{100.0 * summary_df['tap_ready'].mean():.1f}%",
    }
    kpi_cards = "".join([f"<div class='kpi-card'><div class='kpi-title'>{k}</div><div class='kpi-value'>{v}</div></div>" for k, v in kpis.items()])

    figure_labels = {
        "reward_mean_std.png": "Figure 1. Mean reward by evaluated policy",
        "tap_ready_rate.png": "Figure 2. Tap ready rate by evaluated policy",
        "energy_per_ton.png": "Figure 3. Energy per ton by evaluated policy",
        "violation_count.png": "Figure 4. Violation count by evaluated policy",
        "normalized_score_ranking.png": "Figure 5. Normalized score by evaluated policy",
        "scenario_policy_heatmap.png": "Figure 6. Scenario-policy normalized score heatmap",
        "pareto_reward_vs_energy_per_ton.png": "Figure 7. Pareto frontier: reward vs energy per ton",
        "temperature_trajectory_comparison.png": "Figure 8. Base case temperature trajectories (seed 0)",
    }
    figures_html = "".join(
        [
            f"<div><img src='figures/{f}'><div style='font-size:12px;color:#42526e;margin-bottom:10px'>{figure_labels.get(f, f)}</div></div>"
            for f in figures
        ]
    )
    score_equation = "normalized_score = 0.25*tap_ready_norm + 0.20*production_norm + 0.20*production_efficiency_norm + 0.15*energy_per_ton_norm + 0.10*quality_norm + 0.10*safety_norm"
    score_cols = ["policy", "normalized_score", "tap_ready_norm", "production_norm", "production_efficiency_norm", "energy_per_ton_norm", "quality_norm", "safety_norm"]
    score_components_html = _format_table(policy_stats[score_cols])
    html = f"""
<html>
<head>
<meta charset='utf-8'>
<title>EAF Benchmark Report ({policy_list})</title>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #1a1f36; }}
.panel {{ background: #fff; border-radius: 12px; padding: 18px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.kpi-grid {{ display:grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap:12px; }}
.kpi-card {{ background:#eef3ff; border:1px solid #dbe6ff; border-radius:10px; padding:12px; }}
.kpi-title {{ font-size:12px; color:#42526e; }}
.kpi-value {{ font-size:22px; font-weight:700; color:#0f2454; margin-top:4px; }}
.styled-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.styled-table th, .styled-table td {{ border: 1px solid #e6ebf5; padding: 8px 10px; text-align: left; }}
.styled-table th {{ background: #f0f4ff; }}
.proposed {{ color:#b22222; font-weight:700; }}
.warning {{ background:#fff7e6; border:1px solid #ffd591; border-radius:10px; padding:10px; }}
img {{ max-width: 1000px; width: 100%; border-radius: 10px; border: 1px solid #e6ebf5; margin-bottom: 12px; }}
</style>
</head>
<body>
<div class='panel'>
<h1>EAF Benchmark Report: {policy_list}</h1>
<p>All policies are evaluated on Model C enhanced hybrid simulator.</p>
<p>Simulation budget: {max_steps} steps, dt_s = {dt_s} sec, equivalent to {max_steps*dt_s/60:.1f} simulated minutes (expected default: 61.0 min).</p>
<p>Policies: {policy_list}</p>
</div>
<div class='panel'><h2>Balanced Multi-Objective Selection Rule</h2><p><code>{score_equation}</code></p><p>Weights: 0.25 tap_ready, 0.20 production, 0.20 production_efficiency, 0.15 energy_per_ton, 0.10 quality, 0.10 safety.</p><p>Best policy is selected by balanced normalized_score, not by raw reward only.</p></div>
<div class='panel'><h2>KPI Cards</h2><div class='kpi-grid'>{kpi_cards}</div></div>
<div class='panel'><h2>Policy Coverage</h2>{_format_table(policy_coverage)}</div>
<div class='panel warning'><h2>Diagnostic Warnings</h2><ul>{warning_html}</ul></div>
<div class='panel'><h2>Main result table (mean ± std)</h2>{_format_table(policy_stats)}</div>
<div class='panel'><h2>Normalized score components by policy</h2>{score_components_html}</div>
<div class='panel'><h2>Scenario-level ranking table</h2>{_format_table(scenario_rank)}</div>
<div class='panel'><h2>Baseline comparison table</h2>{_format_table(comparison_df)}</div>
<div class='panel'><h2>Statistical significance table</h2>{_format_table(stat_tests)}</div>
<div class='panel'><h2>Best policy decision</h2><p>Best policy by normalized_score: <b>{_display_name(str(best['policy']))}</b> (score={best['normalized_score']:.4f}).</p></div>
<div class='panel'><h2>Figures (all evaluated policies)</h2>{figures_html}</div>
</body>
</html>
"""
    (output_dir / "result.html").write_text(html)


def _build_policies(args: argparse.Namespace) -> tuple[dict[str, BasePolicy], list[str], dict[str, str], list[str]]:
    policies: dict[str, BasePolicy] = {
        "baseline_schedule": IndustrialBaselineSchedulePolicy(),
        "rule_based": RuleBasedPolicy(),
        "mpc": MPCPolicy(horizon=args.mpc_horizon),
    }
    missing_required: list[str] = []
    checkpoint_status: dict[str, str] = {}
    checkpoint_warnings: list[str] = []

    tac_path = args.training_dir / "trainable_adaptive_controller" / "best_policy.json"
    if tac_path.exists():
        policies["trainable_adaptive_controller"] = TrainablePolicy.load(tac_path)
        checkpoint_status["trainable_adaptive_controller"] = "loaded"
    else:
        missing_required.append("trainable_adaptive_controller")
        checkpoint_status["trainable_adaptive_controller"] = "missing checkpoint"

    if args.include_rl_baselines:
        ckpts = {
            "q_learning": (args.training_dir / "q_learning" / "q_table.json", QLearningPolicy.load),
            "dqn": (args.training_dir / "dqn" / "best_policy.npy", DQNPolicy.load),
            "ppo": (args.training_dir / "ppo" / "best_policy.pt", PPOPolicy.load),
            "behavior_cloning": (args.training_dir / "behavior_cloning" / "policy.json", BehaviorCloningPolicy.load),
            "safe_ppo_agentic_mpc": (args.training_dir / "safe_ppo_agentic_mpc" / "best_safe_ppo_agentic_mpc_policy.pt", lambda path: SafePPOAgenticMPCPolicy.load(path, horizon=args.mpc_horizon)),
            "safe_ppo_agentic_sac": (args.training_dir / "safe_ppo_agentic_sac" / "best_safe_ppo_agentic_sac_policy.pt", SafePPOAgenticSACPolicy.load),
            "safe_ppo_agentic_td3": (args.training_dir / "safe_ppo_agentic_td3" / "best_safe_ppo_agentic_td3_policy.pt", SafePPOAgenticTD3Policy.load),
        }
        for name, (path, loader) in ckpts.items():
            if path.exists():
                try:
                    loaded = loader(path)
                    if name == "dqn" and getattr(loaded, "weights", np.zeros((1, 1))).shape[1] != 13:
                        checkpoint_status[name] = "checkpoint architecture mismatch"
                        checkpoint_warnings.append(f"DQN checkpoint architecture mismatch at {path}")
                    else:
                        policies[name] = loaded
                        checkpoint_status[name] = "loaded"
                except Exception as exc:
                    checkpoint_status[name] = f"checkpoint load error: {type(exc).__name__}"
                    checkpoint_warnings.append(f"{name} checkpoint load failed ({type(exc).__name__}) at {path}")
                    missing_required.append(name)
            else:
                checkpoint_status[name] = "missing checkpoint"
                missing_required.append(name)
        policies["sac_inspired"] = SACInspiredPolicy()
        policies["td3_inspired"] = TD3InspiredPolicy()

    if missing_required and not args.allow_missing_rl_baselines:
        raise ValueError(
            "Missing required policy implementations/checkpoints: "
            + ", ".join(missing_required)
            + ". Pass --allow-missing-rl-baselines to continue."
        )
    return policies, missing_required, checkpoint_status, checkpoint_warnings


def _write_tap_diagnostics(summary_df: pd.DataFrame, output_dir: Path) -> None:
    cols = [
        "seed",
        "scenario",
        "policy",
        "reached_tap_temp",
        "max_bath_temp_c",
        "final_bath_temp_c",
        "max_melted_fraction",
        "can_tap_ever_true",
        "tap_command_ever_true",
        "tap_blocked_by_safety_filter_count",
        "termination_reason",
    ]
    diag = summary_df[cols].copy()
    diag.to_csv(output_dir / "tap_diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained/default agent benchmark over EAF scenarios")
    parser.add_argument("--config", type=Path, default=Path("configs/base_case.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/agent_run"))
    parser.add_argument("--training-dir", type=Path, default=Path("results/agent_training"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n-scenarios", type=int, default=6)
    parser.add_argument("--model", choices=["C"], default="C")
    parser.add_argument("--mpc-horizon", type=int, default=8)
    parser.add_argument("--report-format", default="html,csv,md")
    parser.add_argument("--max-steps", type=int, default=610)
    parser.add_argument("--include-rl-baselines", action="store_true")
    parser.add_argument("--allow-missing-rl-baselines", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policies, missing_required, checkpoint_status, checkpoint_warnings = _build_policies(args)
    evaluated_policies = list(policies.keys())

    seeds = list(range(args.seeds))
    scenario_order = ["base_case", "higher_oxygen", "higher_natural_gas", "improved_foamy_slag", "dri20", "delayed_melting_downtime"][: args.n_scenarios]
    summary_df = run_benchmark(args.config, policies, output_dir, seeds=seeds, selected_scenarios=scenario_order, max_steps=args.max_steps)

    missing_from_results = sorted(set(evaluated_policies) - set(summary_df["policy"].unique()))
    if missing_from_results and not args.allow_missing_rl_baselines:
        raise ValueError(f"Missing policies in benchmark output: {missing_from_results}. Pass --allow-missing-rl-baselines to continue.")

    summary_df["normalized_score"] = _normalized_score(summary_df)
    summary_df = summary_df.sort_values(["scenario", "total_reward"], ascending=[True, False]).reset_index(drop=True)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    policy_stats = _policy_stats(summary_df)
    policy_stats["overall_rank"] = np.arange(1, len(policy_stats) + 1)
    policy_stats.to_csv(output_dir / "policy_aggregate_summary.csv", index=False)

    summary_df["rank_by_scenario"] = summary_df.groupby(["seed", "scenario"])["normalized_score"].rank(ascending=False, method="dense")
    scenario_rank = summary_df.groupby(["scenario", "policy"], as_index=False)["rank_by_scenario"].mean().sort_values(["scenario", "rank_by_scenario"])

    kpi_comparison = summary_df.pivot_table(index=["seed", "scenario"], columns="policy", values=["cum_tapped_kg", "total_reward", "cum_electric_mwh", "cum_oxygen_nm3", "cum_ng_nm3", "final_temp_c", "normalized_score"])
    kpi_comparison = kpi_comparison.reindex(columns=evaluated_policies, level=1)
    kpi_comparison.to_csv(output_dir / "kpi_comparison.csv")

    baseline = summary_df[summary_df["policy"] == "baseline_schedule"][["seed", "scenario", "total_reward", "cum_tapped_kg", "cum_electric_mwh"]]
    comparison_rows = []
    for policy in evaluated_policies:
        if policy == "baseline_schedule":
            continue
        cur = summary_df[summary_df["policy"] == policy][["seed", "scenario", "total_reward", "cum_tapped_kg", "cum_electric_mwh"]]
        j = cur.merge(baseline, on=["seed", "scenario"], suffixes=("_policy", "_baseline"))
        if j.empty:
            continue
        reward_gain = _safe_pct(j["total_reward_policy"] - j["total_reward_baseline"], j["total_reward_baseline"].abs())
        comparison_rows.append(
            {
                "policy": policy,
                "reward_delta": float((j["total_reward_policy"] - j["total_reward_baseline"]).mean()),
                "tapped_delta_kg": float((j["cum_tapped_kg_policy"] - j["cum_tapped_kg_baseline"]).mean()),
                "electric_delta_mwh": float((j["cum_electric_mwh_policy"] - j["cum_electric_mwh_baseline"]).mean()),
                "reward_gain_vs_baseline_pct": "n/a" if (reward_gain == "n/a").any() else float(pd.to_numeric(reward_gain).mean()),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    stat_df = summary_df.groupby("policy", as_index=False).agg(mean=("total_reward", "mean"), std=("total_reward", "std"), median=("total_reward", "median"), min=("total_reward", "min"), max=("total_reward", "max"))
    stat_df.to_csv(output_dir / "statistical_analysis.csv", index=False)

    pairs = []
    baselines_for_tests = [p for p in ["baseline_schedule", "mpc", "ppo"] if p in evaluated_policies]
    for target in evaluated_policies:
        for baseline_name in baselines_for_tests:
            if target == baseline_name:
                continue
            r = _paired_stats(summary_df, baseline_name, target)
            if r:
                pairs.append(r)
    stat_tests = pd.DataFrame(pairs)
    stat_tests.to_csv(output_dir / "statistical_tests.csv", index=False)

    policy_coverage = _build_policy_coverage(summary_df, evaluated_policies)
    policy_coverage["checkpoint_status"] = policy_coverage["policy"].map(checkpoint_status).fillna("n/a")
    policy_coverage.to_csv(output_dir / "policy_coverage.csv", index=False)

    _write_tap_diagnostics(summary_df, output_dir)

    warnings: list[str] = []
    if missing_required:
        warnings.append(f"Missing required policies/checkpoints (allowed): {', '.join(missing_required)}")
    warnings.extend(checkpoint_warnings)
    dt_s = float(load_config(args.config).dt_s)
    simulated_minutes = args.max_steps * dt_s / 60.0
    tap_window_start_min = min(60.0, float(load_config(args.config).heat_duration_min) - 5.0)
    if simulated_minutes < tap_window_start_min:
        warnings.append(
            f"Simulation horizon ({simulated_minutes:.1f} min) ends before tap window starts ({tap_window_start_min:.1f} min). "
            "Tap metrics and energy_per_ton/Pareto charts may be invalid."
        )
    no_tap = summary_df[summary_df["tap_ready"] == False]["policy"].unique().tolist()  # noqa: E712
    if no_tap:
        warnings.append("No successful taps for: " + ", ".join(_display_name(p) for p in sorted(no_tap)))

    figures = _plot_all_figures(summary_df, output_dir, evaluated_policies)
    _render_html(output_dir, summary_df, policy_stats, scenario_rank, comparison_df, stat_tests, policy_coverage, figures, args.max_steps, dt_s, warnings)

    (output_dir / "report.md").write_text("# Agent Run Report\n\nSee result.html and CSV artifacts for full benchmark.")
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "policies": evaluated_policies,
                "evaluated_policies": evaluated_policies,
                "policy_display_names": {k: _display_name(k) for k in evaluated_policies},
                "config": str(args.config),
                "model_name": "Model_C_enhanced_hybrid",
                "max_steps": args.max_steps,
                "dt_s": dt_s,
                "simulated_minutes": args.max_steps * dt_s / 60.0,
                "early_termination_allowed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
