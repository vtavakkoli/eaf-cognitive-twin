from __future__ import annotations

import tempfile
from pathlib import Path

from agents.base import BasePolicy
from agents.policies.baseline_schedule import IndustrialBaselineSchedulePolicy
from agents.policies.rule_based import RuleBasedPolicy
from agents.runners.benchmark_runner import run_benchmark


class NeverTapPolicy(BasePolicy):
    name = "never_tap"

    def act(self, observation):
        return {
            "power_mw": 0.0,
            "oxygen_nm3_min": 0.0,
            "ng_nm3_min": 0.0,
            "carbon_kg_min": 0.0,
            "flux_kg_min": 0.0,
            "tap_command": False,
        }


def test_benchmark_max_steps_and_csv_outputs():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        df = run_benchmark(Path("configs/base_case.json"), {"baseline_schedule": IndustrialBaselineSchedulePolicy(), "rule_based": RuleBasedPolicy()}, out, seeds=[0], selected_scenarios=["base_case"], max_steps=610)
        assert not df.empty
        assert (out / "timeseries").exists()


def test_energy_per_ton_is_nan_when_no_tapping():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        df = run_benchmark(Path("configs/base_case.json"), {"never_tap": NeverTapPolicy()}, out, seeds=[0], selected_scenarios=["base_case"], max_steps=20)
        assert df["cum_tapped_kg"].iloc[0] == 0.0
        assert df["energy_per_ton"].isna().iloc[0]
