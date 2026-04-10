from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from agents.policies.mpc_policy import MPCPolicy
from agents.policies.rule_based import RuleBasedPolicy
from agents.policies.trainable_policy import TrainablePolicy
from agents.runners.benchmark_runner import run_benchmark


def _plot_summary(summary_df, output_dir: Path) -> None:
    for metric in ["cum_tapped_kg", "final_temp_c", "cum_electric_mwh", "cum_oxygen_nm3", "cum_ng_nm3", "final_carbon_wt_pct"]:
        pivot = summary_df.pivot(index="scenario", columns="policy", values=metric)
        ax = pivot.plot(kind="bar", figsize=(10, 4), title=f"{metric} by scenario/policy")
        ax.set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f"plot_{metric}_comparison.png", dpi=150)
        plt.close()


def _plot_report_visuals(summary_df: pd.DataFrame, policy_stats: pd.DataFrame, output_dir: Path) -> list[str]:
    generated: list[str] = []
    # 1) Composite score chart.
    metrics = [
        ("mean_reward", True),
        ("mean_tapped_kg", True),
        ("mean_electric_mwh", False),
        ("mean_oxygen_nm3", False),
        ("mean_ng_nm3", False),
    ]
    score_df = policy_stats.copy()
    for col, higher_is_better in metrics:
        vals = score_df[col]
        lo, hi = float(vals.min()), float(vals.max())
        if abs(hi - lo) < 1e-9:
            score_df[f"{col}_norm"] = 0.5
        elif higher_is_better:
            score_df[f"{col}_norm"] = (vals - lo) / (hi - lo)
        else:
            score_df[f"{col}_norm"] = (hi - vals) / (hi - lo)
    norm_cols = [f"{m[0]}_norm" for m in metrics]
    score_df["composite_score"] = score_df[norm_cols].mean(axis=1)
    ax = score_df.sort_values("composite_score", ascending=False).plot(
        x="policy", y="composite_score", kind="bar", color="#4C78A8", legend=False, figsize=(8, 4), title="Composite policy score (higher is better)"
    )
    ax.set_ylabel("normalized score")
    plt.tight_layout()
    fname = "plot_composite_policy_score.png"
    plt.savefig(output_dir / fname, dpi=160)
    plt.close()
    generated.append(fname)

    # 2) Scenario heatmap (reward by policy/scenario).
    reward_pivot = summary_df.pivot(index="scenario", columns="policy", values="total_reward")
    plt.figure(figsize=(9, 4.5))
    im = plt.imshow(reward_pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="total_reward")
    plt.xticks(range(len(reward_pivot.columns)), reward_pivot.columns, rotation=30, ha="right")
    plt.yticks(range(len(reward_pivot.index)), reward_pivot.index)
    plt.title("Scenario reward heatmap")
    plt.tight_layout()
    fname = "plot_reward_heatmap.png"
    plt.savefig(output_dir / fname, dpi=160)
    plt.close()
    generated.append(fname)
    return generated


def _plot_scenario_trajectory_visuals(output_dir: Path, policies: list[str], scenarios: list[str]) -> list[str]:
    generated: list[str] = []
    timeseries_dir = output_dir / "timeseries"
    if not timeseries_dir.exists():
        return generated

    for policy in policies:
        scenario_frames: dict[str, pd.DataFrame] = {}
        for scenario in scenarios:
            ts_path = timeseries_dir / f"agent_timeseries_{scenario}_{policy}.csv"
            if ts_path.exists():
                scenario_frames[scenario] = pd.read_csv(ts_path)
        if not scenario_frames:
            continue

        # Energy profile comparison: cumulative electric energy by scenario for one policy.
        fig, ax = plt.subplots(figsize=(10, 5))
        for scenario, frame in scenario_frames.items():
            if "time_min" in frame.columns and "cum_electric_mwh" in frame.columns:
                ax.plot(frame["time_min"], frame["cum_electric_mwh"], linewidth=1.8, label=scenario)
        ax.set_title(f"Cumulative electric energy profile by scenario ({policy})")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Cumulative electric energy [MWh]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        energy_name = f"plot_energy_profile_all_scenarios_{policy}.png"
        plt.savefig(output_dir / energy_name, dpi=160)
        plt.close()
        generated.append(energy_name)

        # Temperature trajectory comparison: bath temperature by scenario for one policy.
        fig, ax = plt.subplots(figsize=(10, 5))
        for scenario, frame in scenario_frames.items():
            if "time_min" in frame.columns and "bath_temp_c" in frame.columns:
                ax.plot(frame["time_min"], frame["bath_temp_c"], linewidth=1.8, label=scenario)
        ax.set_title(f"Temperature trajectory by scenario ({policy})")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Bath temperature [°C]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plt.tight_layout()
        temp_name = f"plot_temperature_trajectory_all_scenarios_{policy}.png"
        plt.savefig(output_dir / temp_name, dpi=160)
        plt.close()
        generated.append(temp_name)

    return generated


def _policy_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    agg = summary_df.groupby("policy", as_index=False).agg(
        mean_reward=("total_reward", "mean"),
        mean_tapped_kg=("cum_tapped_kg", "mean"),
        mean_electric_mwh=("cum_electric_mwh", "mean"),
        mean_oxygen_nm3=("cum_oxygen_nm3", "mean"),
        mean_ng_nm3=("cum_ng_nm3", "mean"),
        mean_temp_c=("final_temp_c", "mean"),
    )
    return agg.sort_values("mean_reward", ascending=False)


def _paired_stats(summary_df: pd.DataFrame, left: str, right: str, metric: str, higher_is_better: bool) -> dict:
    piv = summary_df.pivot(index="scenario", columns="policy", values=metric).dropna()
    if left not in piv.columns or right not in piv.columns or len(piv) < 2:
        return {"metric": metric, "comparison": f"{left} vs {right}", "mean_delta": float("nan"), "p_value": float("nan")}
    delta = piv[left] - piv[right]
    if not higher_is_better:
        delta = -delta
    t_res = stats.ttest_rel(piv[left], piv[right])
    return {
        "metric": metric,
        "comparison": f"{left} vs {right}",
        "mean_delta": float(delta.mean()),
        "p_value": float(t_res.pvalue),
    }


def _write_result_html(summary_df: pd.DataFrame, output_dir: Path, policy_stats: pd.DataFrame, stat_rows: list[dict], chart_files: list[str]) -> None:
    baseline = summary_df[summary_df["policy"] == "baseline_schedule"].copy()
    baseline = baseline.set_index("scenario")

    comparisons = []
    for policy in sorted(summary_df["policy"].unique()):
        if policy == "baseline_schedule":
            continue
        cur = summary_df[summary_df["policy"] == policy].set_index("scenario")
        joint = cur.join(baseline, lsuffix="_policy", rsuffix="_baseline", how="inner")
        if joint.empty:
            continue
        comparisons.append(
            {
                "policy": policy,
                "reward_gain_vs_baseline_pct": float(
                    ((joint["total_reward_policy"] - joint["total_reward_baseline"]) / joint["total_reward_baseline"].abs().clip(lower=1e-9)).mean() * 100.0
                ),
                "tapped_gain_vs_baseline_pct": float(
                    ((joint["cum_tapped_kg_policy"] - joint["cum_tapped_kg_baseline"]) / joint["cum_tapped_kg_baseline"].abs().clip(lower=1e-9)).mean() * 100.0
                ),
                "electric_reduction_vs_baseline_pct": float(
                    ((joint["cum_electric_mwh_baseline"] - joint["cum_electric_mwh_policy"]) / joint["cum_electric_mwh_baseline"].abs().clip(lower=1e-9)).mean() * 100.0
                ),
            }
        )
    comparison_df = pd.DataFrame(comparisons)
    statistical_df = pd.DataFrame(stat_rows)
    images_html = "".join(
        f'<div><img src="{name}" style="max-width: 960px; width: 100%; border: 1px solid #ddd; margin-bottom: 16px;"/></div>'
        for name in chart_files
    )

    winner = policy_stats.iloc[0]["policy"] if not policy_stats.empty else "n/a"
    html = f"""
<html>
<head>
  <meta charset="utf-8" />
  <title>Agent Benchmark Result</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Agent Benchmark Result</h1>
  <p>Best policy by mean reward: <strong>{winner}</strong></p>
  <h2>Scenario-level KPI Summary</h2>
  {summary_df.to_html(index=False, float_format=lambda x: f"{x:,.4f}")}
  <h2>Policy-level Aggregated KPIs</h2>
  {policy_stats.to_html(index=False, float_format=lambda x: f"{x:,.4f}")}
  <h2>Baseline Comparison (% improvements)</h2>
  {comparison_df.to_html(index=False, float_format=lambda x: f"{x:,.4f}") if not comparison_df.empty else '<p>No comparable policies found.</p>'}
  <h2>Statistical Analysis (paired t-tests across scenarios)</h2>
  <p>Positive mean_delta indicates better performance under the metric orientation.</p>
  {statistical_df.to_html(index=False, float_format=lambda x: f"{x:,.6f}") if not statistical_df.empty else '<p>Insufficient data for significance testing.</p>'}
  <h2>Key visuals</h2>
  {images_html}
</body>
</html>
"""
    (output_dir / "result.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trained/default agent benchmark over EAF scenarios")
    parser.add_argument("--config", type=Path, default=Path("configs/base_case.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/agent_run"))
    parser.add_argument("--trained-policy", type=Path, default=Path("results/agent_training/checkpoints/best_policy.json"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = {
        "baseline_schedule": RuleBasedPolicy(),
        "rule_based": RuleBasedPolicy(),
        "mpc": MPCPolicy(),
    }
    if args.trained_policy.exists():
        policies["agentic_ai"] = TrainablePolicy.load(args.trained_policy)

    summary_df = run_benchmark(args.config, policies, output_dir)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)
    policy_stats = _policy_stats(summary_df)
    policy_stats.to_csv(output_dir / "policy_aggregate_summary.csv", index=False)

    kpi_comparison = summary_df.pivot_table(
        index=["scenario"],
        columns="policy",
        values=["cum_tapped_kg", "total_reward", "cum_electric_mwh", "cum_oxygen_nm3", "cum_ng_nm3", "final_temp_c"],
    )
    kpi_comparison.to_csv(output_dir / "kpi_comparison.csv")

    _plot_summary(summary_df, output_dir)
    chart_files = _plot_report_visuals(summary_df, policy_stats, output_dir)
    chart_files.extend(
        _plot_scenario_trajectory_visuals(
            output_dir=output_dir,
            policies=sorted(summary_df["policy"].unique()),
            scenarios=sorted(summary_df["scenario"].unique()),
        )
    )
    stat_rows = []
    for metric, higher_is_better in [
        ("total_reward", True),
        ("cum_tapped_kg", True),
        ("cum_electric_mwh", False),
        ("cum_oxygen_nm3", False),
        ("cum_ng_nm3", False),
    ]:
        available_policies = set(summary_df["policy"])
        if "agentic_ai" in available_policies:
            stat_rows.append(_paired_stats(summary_df, "agentic_ai", "baseline_schedule", metric, higher_is_better))
            stat_rows.append(_paired_stats(summary_df, "agentic_ai", "rule_based", metric, higher_is_better))
            stat_rows.append(_paired_stats(summary_df, "agentic_ai", "mpc", metric, higher_is_better))
        if "mpc" in available_policies:
            stat_rows.append(_paired_stats(summary_df, "mpc", "baseline_schedule", metric, higher_is_better))
            stat_rows.append(_paired_stats(summary_df, "mpc", "rule_based", metric, higher_is_better))
        stat_rows.append(_paired_stats(summary_df, "rule_based", "baseline_schedule", metric, higher_is_better))
    pd.DataFrame(stat_rows).to_csv(output_dir / "statistical_analysis.csv", index=False)
    _write_result_html(summary_df, output_dir, policy_stats, stat_rows, chart_files)

    report_lines = [
        "# Agent Run Report",
        "",
        "## Policies",
        *(f"- {name}" for name in policies.keys()),
        "",
        "## Output files",
        "- scenario_summary.csv",
        "- policy_aggregate_summary.csv",
        "- kpi_comparison.csv",
        "- statistical_analysis.csv",
        "- result.html",
        "- timeseries/*.csv",
        "- plot_*_comparison.png",
        "- plot_composite_policy_score.png",
        "- plot_reward_heatmap.png",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines))
    (output_dir / "run_manifest.json").write_text(json.dumps({"policies": list(policies.keys()), "config": str(args.config)}, indent=2))


if __name__ == "__main__":
    main()
