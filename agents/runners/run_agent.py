from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

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
    }
    if args.trained_policy.exists():
        policies["trained"] = TrainablePolicy.load(args.trained_policy)

    summary_df = run_benchmark(args.config, policies, output_dir)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    kpi_comparison = summary_df.pivot_table(
        index=["scenario"],
        columns="policy",
        values=["cum_tapped_kg", "total_reward", "cum_electric_mwh", "cum_oxygen_nm3", "cum_ng_nm3", "final_temp_c"],
    )
    kpi_comparison.to_csv(output_dir / "kpi_comparison.csv")

    _plot_summary(summary_df, output_dir)

    report_lines = [
        "# Agent Run Report",
        "",
        "## Policies",
        *(f"- {name}" for name in policies.keys()),
        "",
        "## Output files",
        "- scenario_summary.csv",
        "- kpi_comparison.csv",
        "- timeseries/*.csv",
        "- plot_*_comparison.png",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines))
    (output_dir / "run_manifest.json").write_text(json.dumps({"policies": list(policies.keys()), "config": str(args.config)}, indent=2))


if __name__ == "__main__":
    main()
