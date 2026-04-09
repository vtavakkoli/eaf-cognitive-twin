from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
import os
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eaf_twin.config.loader import load_config

from agents.controller import EAFController
from agents.policies.rule_based import RuleBasedPolicy


class TestAgentController(unittest.TestCase):
    def test_controller_reset_step(self):
        ctrl = EAFController(load_config(Path("configs/base_case.json")))
        obs = ctrl.reset()
        self.assertIn("time_min", obs)
        res = ctrl.step({"power_mw": 80, "oxygen_nm3_min": 60, "ng_nm3_min": 12, "carbon_kg_min": 15, "flux_kg_min": 90, "tap_command": False})
        self.assertIn("melted_fraction", res.observation)

    def test_rule_based_episode_progress(self):
        ctrl = EAFController(load_config(Path("configs/base_case.json")))
        policy = RuleBasedPolicy()
        obs = ctrl.reset()
        for _ in range(40):
            out = ctrl.step(policy.act(obs))
            obs = out.observation
            if out.done:
                break
        self.assertGreaterEqual(float(obs["melted_fraction"]), 0.0)


class TestAgentRunners(unittest.TestCase):
    def test_train_and_run_create_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_out = root / "train"
            run_out = root / "run"
            subprocess.run(
                [sys.executable, "-m", "agents.runners.train_agent", "--config", "configs/base_case.json", "--output-dir", str(train_out), "--iterations", "2"],
                check=True,
                env={**os.environ, "PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"},
            )
            self.assertTrue((train_out / "training_log.csv").exists())
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agents.runners.run_agent",
                    "--config",
                    "configs/base_case.json",
                    "--output-dir",
                    str(run_out),
                    "--trained-policy",
                    str(train_out / "checkpoints" / "best_policy.json"),
                ],
                check=True,
                env={**os.environ, "PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"},
            )
            self.assertTrue((run_out / "scenario_summary.csv").exists())
            self.assertTrue((run_out / "statistical_analysis.csv").exists())
            self.assertTrue((run_out / "result.html").exists())
            summary = pd.read_csv(run_out / "scenario_summary.csv")
            policies = set(summary["policy"].unique())
            self.assertIn("baseline_schedule", policies)
            self.assertIn("rule_based", policies)
            self.assertIn("mpc", policies)
            self.assertIn("agentic_ai", policies)


if __name__ == "__main__":
    unittest.main()
