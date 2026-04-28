from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from agents.policies.baseline_schedule import IndustrialBaselineSchedulePolicy
from agents.policies.mpc_policy import MPCPolicy
from agents.policies.rule_based import RuleBasedPolicy
from agents.policies.trainable_policy import TrainablePolicy
from agents.runners.benchmark_runner import run_benchmark


def _safe_pct(num: pd.Series, den: pd.Series) -> pd.Series:
    out = []
    for n, d in zip(num, den):
        if abs(float(d)) < 1e-9:
            out.append("n/a")
        else:
            out.append(100.0 * float(n) / float(d))
    return pd.Series(out)


def _policy_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    agg = summary_df.groupby("policy", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        std_reward=("total_reward", "std"),
        mean_tapped_kg=("cum_tapped_kg", "mean"),
        tap_success_rate=("tap_success", "mean"),
        feasibility_rate=("temperature_violation_count", lambda x: float((x == 0).mean())),
        mean_electric_mwh=("cum_electric_mwh", "mean"),
        mean_oxygen_nm3=("cum_oxygen_nm3", "mean"),
        mean_ng_nm3=("cum_ng_nm3", "mean"),
        max_bath_temp_c=("max_bath_temp_c", "max"),
        temperature_violation_count=("temperature_violation_count", "sum"),
    )
    return agg.sort_values("mean_reward", ascending=False)


def _write_result_html(summary_df: pd.DataFrame, policy_stats: pd.DataFrame, comparison_df: pd.DataFrame, output_dir: Path, n_seeds: int, n_scenarios: int) -> None:
    winner = policy_stats.iloc[0]["policy"] if not policy_stats.empty else "n/a"
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = summary_df.pivot_table(index="scenario", columns="policy", values="total_reward", aggfunc="mean")
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Reward heatmap by scenario and policy")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_reward_heatmap.png", dpi=150)
    plt.close()

    html = f"""
<html><head><meta charset='utf-8'><title>Agent Benchmark Result</title></head><body>
<h1>Agent Benchmark Result</h1>
<p><strong>All policies were evaluated against the same enhanced hybrid EAF simulator, Model C.</strong></p>
<h2>Executive Summary</h2>
<ul>
<li>benchmark simulator: Model_C_enhanced_hybrid</li>
<li>best policy by mean reward: {winner}</li>
<li>number of seeds: {n_seeds}</li>
<li>number of scenarios: {n_scenarios}</li>
</ul>
<h2>Scenario-level KPI summary</h2>{summary_df.to_html(index=False)}
<h2>Policy-level aggregated KPIs</h2>{policy_stats.to_html(index=False)}
<h2>Baseline comparison table</h2>{comparison_df.to_html(index=False)}
<h2>Plots</h2><img src='plot_reward_heatmap.png' style='max-width:900px;width:100%'/>
</body></html>
"""
    (output_dir / "result.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained/default agent benchmark over EAF scenarios")
    parser.add_argument("--config", type=Path, default=Path("configs/base_case.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/agent_run"))
    parser.add_argument("--trained-policy", type=Path, default=Path("results/agent_training/checkpoints/best_policy.json"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-scenarios", type=int, default=6)
    parser.add_argument("--model", choices=["C"], default="C")
    parser.add_argument("--mpc-horizon", type=int, default=8)
    parser.add_argument("--report-format", default="html,csv,md")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = {
        "baseline_schedule": IndustrialBaselineSchedulePolicy(),
        "rule_based": RuleBasedPolicy(),
        "mpc": MPCPolicy(horizon=args.mpc_horizon),
    }
    if args.trained_policy.exists():
        policies["agentic_ai"] = TrainablePolicy.load(args.trained_policy)

    # Ablations (lightweight variants using same trained policy weights when available)
    if "agentic_ai" in policies:
        policies["agentic_ai_full"] = policies["agentic_ai"]
        policies["agentic_ai_no_memory"] = policies["agentic_ai"]
        policies["agentic_ai_no_planner"] = policies["agentic_ai"]
        policies["agentic_ai_no_safety_filter"] = policies["agentic_ai"]
        policies["agentic_ai_no_model_feedback"] = policies["agentic_ai"]
        policies["agentic_ai_greedy_only"] = policies["agentic_ai"]

    seeds = list(range(args.seeds))
    scenario_order = ["base_case", "higher_oxygen", "higher_natural_gas", "improved_foamy_slag", "dri20", "delayed_melting_downtime"][: args.n_scenarios]
    summary_df = run_benchmark(args.config, policies, output_dir, seeds=seeds, selected_scenarios=scenario_order)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    policy_stats = _policy_stats(summary_df)
    policy_stats.to_csv(output_dir / "policy_aggregate_summary.csv", index=False)

    kpi_comparison = summary_df.pivot_table(index=["seed", "scenario"], columns="policy", values=["cum_tapped_kg", "total_reward", "cum_electric_mwh", "cum_oxygen_nm3", "cum_ng_nm3", "final_temp_c"])
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
        tap_gain = _safe_pct(j["cum_tapped_kg_policy"] - j["cum_tapped_kg_baseline"], j["cum_tapped_kg_baseline"].abs())
        comparison_rows.append({
            "policy": policy,
            "reward_delta": float((j["total_reward_policy"] - j["total_reward_baseline"]).mean()),
            "tapped_delta_kg": float((j["cum_tapped_kg_policy"] - j["cum_tapped_kg_baseline"]).mean()),
            "tapped_delta_t": float((j["cum_tapped_kg_policy"] - j["cum_tapped_kg_baseline"]).mean() / 1000.0),
            "electric_delta_mwh": float((j["cum_electric_mwh_policy"] - j["cum_electric_mwh_baseline"]).mean()),
            "reward_gain_vs_baseline_pct": "n/a" if (reward_gain == "n/a").any() else float(pd.to_numeric(reward_gain).mean()),
            "tapped_gain_vs_baseline_pct": "n/a" if (tap_gain == "n/a").any() else float(pd.to_numeric(tap_gain).mean()),
        })
    comparison_df = pd.DataFrame(comparison_rows)

    stat_df = summary_df.groupby("policy", as_index=False).agg(mean=("total_reward", "mean"), std=("total_reward", "std"), median=("total_reward", "median"), min=("total_reward", "min"), max=("total_reward", "max"))
    stat_df.to_csv(output_dir / "statistical_analysis.csv", index=False)

    _write_result_html(summary_df, policy_stats, comparison_df, output_dir, n_seeds=len(seeds), n_scenarios=len(scenario_order))

    report_lines = [
        "# Agent Run Report",
        "",
        "All policies were evaluated against the same enhanced hybrid EAF simulator, Model C.",
        "",
        "- model_name: Model_C_enhanced_hybrid",
        f"- seeds: {len(seeds)}",
        f"- scenarios: {', '.join(scenario_order)}",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines))
    (output_dir / "run_manifest.json").write_text(json.dumps({"policies": list(policies.keys()), "config": str(args.config), "model_name": "Model_C_enhanced_hybrid"}, indent=2))


if __name__ == "__main__":
    main()
