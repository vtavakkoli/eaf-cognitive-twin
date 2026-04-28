from __future__ import annotations

import tempfile
from pathlib import Path

from agents.policies.baseline_schedule import IndustrialBaselineSchedulePolicy
from agents.policies.rule_based import RuleBasedPolicy
from agents.runners.benchmark_runner import run_benchmark


def test_benchmark_max_steps_and_csv_outputs():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        df = run_benchmark(Path("configs/base_case.json"), {"baseline_schedule": IndustrialBaselineSchedulePolicy(), "rule_based": RuleBasedPolicy()}, out, seeds=[0], selected_scenarios=["base_case"], max_steps=650)
        assert not df.empty
        assert (out / "timeseries").exists()
