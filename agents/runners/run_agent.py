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
    return (
        summary_df.groupby("policy", as_index=False)
        .agg(
            mean_reward=("total_reward", "mean"),
            std_reward=("total_reward", "std"),
            mean_tapped_kg=("cum_tapped_kg", "mean"),
            tap_success_rate=("tap_success", "mean"),
            feasibility_rate=("temperature_violation_count", lambda x: float((x == 0).mean())),
            mean_electric_mwh=("cum_electric_mwh", "mean"),
            mean_oxygen_nm3=("cum_oxygen_nm3", "mean"),
            mean_ng_nm3=("cum_ng_nm3", "mean"),
            mean_heat_steps=("steps", "mean"),
            max_bath_temp_c=("max_bath_temp_c", "max"),
            temperature_violation_count=("temperature_violation_count", "sum"),
            safety_violation_count=("safety_violation_count", "sum"),
            invalid_tap_count=("invalid_tap_count", "sum"),
            action_clamp_count=("action_clamp_count", "sum"),
        )
        .sort_values("mean_reward", ascending=False)
    )


def _plot_temperature_trajectories(output_dir: Path, scenarios: list[str]) -> list[str]:
    fig_dir = output_dir / "figures"
    ts_dir = output_dir / "timeseries"
    files: list[str] = []

    for scenario in scenarios:
        scenario_frames: list[pd.DataFrame] = []
        pattern = f"agent_timeseries_{scenario}_*_seed*.csv"
        for ts_path in sorted(ts_dir.glob(pattern)):
            stem = ts_path.stem
            prefix = f"agent_timeseries_{scenario}_"
            if not stem.startswith(prefix) or "_seed" not in stem:
                continue
            policy_name = stem[len(prefix) : stem.rfind("_seed")]
            ts_df = pd.read_csv(ts_path)
            if "time_min" not in ts_df.columns or "bath_temp_c" not in ts_df.columns:
                continue
            scenario_frames.append(ts_df[["time_min", "bath_temp_c"]].assign(policy=policy_name))

        if not scenario_frames:
            continue

        merged = pd.concat(scenario_frames, ignore_index=True)
        curve_df = (
            merged.groupby(["policy", "time_min"], as_index=False)["bath_temp_c"]
            .mean()
            .sort_values(["policy", "time_min"])
        )
        plt.figure(figsize=(9.5, 4.5))
        for policy_name, policy_df in curve_df.groupby("policy"):
            plt.plot(policy_df["time_min"], policy_df["bath_temp_c"], label=policy_name, linewidth=2)
        plt.xlabel("time_min")
        plt.ylabel("bath_temp_c")
        plt.title(f"Temperature trajectory by policy ({scenario})")
        plt.legend()
        plt.tight_layout()
        figure_name = f"temperature_trajectory_{scenario}.png"
        plt.savefig(fig_dir / figure_name, dpi=160)
        plt.close()
        files.append(figure_name)

    return files


def _plot_all_figures(summary_df: pd.DataFrame, output_dir: Path, scenarios: list[str]) -> list[str]:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    # reward heatmap by scenario/policy
    pivot = summary_df.pivot_table(index="scenario", columns="policy", values="total_reward", aggfunc="mean")
    plt.figure(figsize=(10, 4.5))
    im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="mean reward")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Reward heatmap by scenario and policy")
    plt.tight_layout()
    f = "reward_heatmap.png"
    plt.savefig(fig_dir / f, dpi=160)
    plt.close()
    files.append(f)

    policy_agg = summary_df.groupby("policy", as_index=False).agg(
        tap_success_rate=("tap_success", "mean"),
        max_bath_temp_c=("max_bath_temp_c", "max"),
        temperature_violation_count=("temperature_violation_count", "sum"),
        mean_tapped_kg=("cum_tapped_kg", "mean"),
        mean_electric_mwh=("cum_electric_mwh", "mean"),
        mean_oxygen_nm3=("cum_oxygen_nm3", "mean"),
        mean_ng_nm3=("cum_ng_nm3", "mean"),
    )

    for col, fname, title, ylabel in [
        ("tap_success_rate", "tap_success_rate.png", "Tap success rate by policy", "rate"),
        ("max_bath_temp_c", "max_bath_temp.png", "Max bath temperature by policy", "°C"),
        ("temperature_violation_count", "temp_violation_count.png", "Temperature violation count by policy", "count"),
    ]:
        ax = policy_agg.sort_values(col, ascending=False).plot(x="policy", y=col, kind="bar", legend=False, figsize=(8, 4), color="#2C7FB8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=160)
        plt.close()
        files.append(fname)

    # resource usage grouped bars
    res = policy_agg.set_index("policy")[["mean_electric_mwh", "mean_oxygen_nm3", "mean_ng_nm3"]]
    ax = res.plot(kind="bar", figsize=(9, 4.5))
    ax.set_title("Resource usage comparison")
    ax.set_ylabel("mean usage")
    plt.tight_layout()
    fname = "resource_usage_comparison.png"
    plt.savefig(fig_dir / fname, dpi=160)
    plt.close()
    files.append(fname)

    # tapped steel distribution
    ax = summary_df.boxplot(column="cum_tapped_kg", by="policy", figsize=(9, 4.5), rot=30)
    ax.set_title("Tapped steel distribution by policy")
    ax.set_ylabel("cum_tapped_kg")
    plt.suptitle("")
    plt.tight_layout()
    fname = "tapped_steel_distribution.png"
    plt.savefig(fig_dir / fname, dpi=160)
    plt.close()
    files.append(fname)

    # Pareto reward vs energy
    plt.figure(figsize=(8, 4.5))
    plt.scatter(summary_df["cum_electric_mwh"], summary_df["total_reward"], alpha=0.6)
    plt.xlabel("cum_electric_mwh")
    plt.ylabel("total_reward")
    plt.title("Pareto view: reward vs electric energy")
    plt.tight_layout()
    fname = "pareto_reward_vs_energy.png"
    plt.savefig(fig_dir / fname, dpi=160)
    plt.close()
    files.append(fname)

    files.extend(_plot_temperature_trajectories(output_dir, scenarios))

    return files


def _render_html(output_dir: Path, summary_df: pd.DataFrame, policy_stats: pd.DataFrame, comparison_df: pd.DataFrame, n_seeds: int, n_scenarios: int, figures: list[str]) -> None:
    winner = policy_stats.iloc[0]["policy"] if not policy_stats.empty else "n/a"

    scenario_sections = []
    for scenario in sorted(summary_df["scenario"].unique()):
        scen_df = summary_df[summary_df["scenario"] == scenario].sort_values("total_reward", ascending=False)
        scenario_sections.append(f"<h3>{scenario}</h3>{scen_df.to_html(index=False, classes='tbl kpi-table', float_format=lambda x: f'{x:,.4f}')}")
    scenarios_html = "\n".join(scenario_sections)

    figs_html = "\n".join([f"<div class='figure'><img src='figures/{f}' alt='{f}'/></div>" for f in figures])

    html = f"""
<html>
<head>
  <meta charset='utf-8' />
  <title>EAF Agent Benchmark Report</title>
  <style>
    body {{ font-family: 'Inter', 'Segoe UI', Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #1f2a37; }}
    .card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 18px; box-shadow: 0 4px 14px rgba(0,0,0,0.08); }}
    h1, h2, h3 {{ margin-top: 0; color: #0f172a; }}
    .badge {{ display: inline-block; background: #e0ecff; color: #1d4ed8; padding: 6px 10px; border-radius: 999px; margin-right: 8px; font-weight: 600; }}
    .tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .tbl th, .tbl td {{ border: 1px solid #d8dee9; padding: 6px 8px; text-align: right; }}
    .tbl th:first-child, .tbl td:first-child {{ text-align: left; }}
    .kpi-table {{ margin-bottom: 16px; }}
    .figure img {{ width: 100%; max-width: 980px; border: 1px solid #cbd5e1; border-radius: 8px; background: white; margin: 8px 0 14px; }}
  </style>
</head>
<body>
  <div class='card'>
    <h1>EAF Agent Benchmark Report</h1>
    <p><strong>All policies were evaluated against the same enhanced hybrid EAF simulator, Model C.</strong></p>
    <span class='badge'>Simulator: Model_C_enhanced_hybrid</span>
    <span class='badge'>Seeds: {n_seeds}</span>
    <span class='badge'>Scenarios: {n_scenarios}</span>
    <span class='badge'>Best policy (mean reward): {winner}</span>
  </div>

  <div class='card'>
    <h2>Policy-level aggregated KPIs</h2>
    {policy_stats.to_html(index=False, classes='tbl', float_format=lambda x: f"{x:,.4f}")}
  </div>

  <div class='card'>
    <h2>Baseline comparison (safe n/a handling)</h2>
    {comparison_df.to_html(index=False, classes='tbl', float_format=lambda x: f"{x:,.4f}") if not comparison_df.empty else '<p>No comparison rows.</p>'}
  </div>

  <div class='card'>
    <h2>Scenario KPI tables (sorted by total reward)</h2>
    {scenarios_html}
  </div>

  <div class='card'>
    <h2>Figures</h2>
    {figs_html}
  </div>
</body>
</html>
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

    seeds = list(range(args.seeds))
    scenario_order = ["base_case", "higher_oxygen", "higher_natural_gas", "improved_foamy_slag", "dri20", "delayed_melting_downtime"][: args.n_scenarios]
    summary_df = run_benchmark(args.config, policies, output_dir, seeds=seeds, selected_scenarios=scenario_order)
    summary_df = summary_df.sort_values(["scenario", "total_reward"], ascending=[True, False]).reset_index(drop=True)
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
        comparison_rows.append(
            {
                "policy": policy,
                "reward_delta": float((j["total_reward_policy"] - j["total_reward_baseline"]).mean()),
                "tapped_delta_kg": float((j["cum_tapped_kg_policy"] - j["cum_tapped_kg_baseline"]).mean()),
                "tapped_delta_t": float((j["cum_tapped_kg_policy"] - j["cum_tapped_kg_baseline"]).mean() / 1000.0),
                "electric_delta_mwh": float((j["cum_electric_mwh_policy"] - j["cum_electric_mwh_baseline"]).mean()),
                "reward_gain_vs_baseline_pct": "n/a" if (reward_gain == "n/a").any() else float(pd.to_numeric(reward_gain).mean()),
                "tapped_gain_vs_baseline_pct": "n/a" if (tap_gain == "n/a").any() else float(pd.to_numeric(tap_gain).mean()),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    stat_df = summary_df.groupby("policy", as_index=False).agg(mean=("total_reward", "mean"), std=("total_reward", "std"), median=("total_reward", "median"), min=("total_reward", "min"), max=("total_reward", "max"))
    stat_df.to_csv(output_dir / "statistical_analysis.csv", index=False)

    figures = _plot_all_figures(summary_df, output_dir, scenario_order)
    _render_html(output_dir, summary_df, policy_stats, comparison_df, len(seeds), len(scenario_order), figures)

    report_lines = [
        "# Agent Run Report",
        "",
        "All policies were evaluated against the same enhanced hybrid EAF simulator, Model C.",
        "",
        "## Scenario KPI tables (sorted by total reward)",
    ]
    for scenario in scenario_order:
        scen = summary_df[summary_df["scenario"] == scenario].sort_values("total_reward", ascending=False)
        report_lines.extend([f"### {scenario}", scen.to_csv(index=False), ""])
    (output_dir / "report.md").write_text("\n".join(report_lines))
    (output_dir / "run_manifest.json").write_text(json.dumps({"policies": list(policies.keys()), "config": str(args.config), "model_name": "Model_C_enhanced_hybrid", "max_steps": 500}, indent=2))


if __name__ == "__main__":
    main()
