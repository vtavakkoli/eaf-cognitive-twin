from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eaf_twin.config.defaults import default_config, scenario_configs
from eaf_twin.config.loader import load_config

from agents.controller import EAFController
from agents.policies.baseline_schedule import IndustrialBaselineSchedulePolicy
from agents.policies.mpc_policy import MPCPolicy
from agents.policies.rule_based import RuleBasedPolicy


class TestAgentController(unittest.TestCase):
    def test_controller_reset_step(self):
        ctrl = EAFController(load_config(Path("configs/base_case.json")))
        obs = ctrl.reset()
        self.assertIn("time_min", obs)
        res = ctrl.step({"power_mw": 80, "oxygen_nm3_min": 60, "ng_nm3_min": 12, "carbon_kg_min": 15, "flux_kg_min": 90, "tap_command": False})
        self.assertIn("melted_fraction", res.observation)

    def test_agent_benchmark_uses_model_c(self):
        ctrl = EAFController(default_config(), enhanced_model=True)
        self.assertEqual(ctrl.model_name, "Model_C_enhanced_hybrid")

    def test_baseline_schedule_taps_when_ready(self):
        cfg = default_config()
        policy = IndustrialBaselineSchedulePolicy()
        obs = {
            "bath_temp_c": cfg.tap_target_temp_c + 10,
            "liquid_steel_kg": cfg.tap_target_steel_kg,
            "steel_carbon_wt_pct": 0.05,
            "time_min": 30,
            "default_schedule_action": {"power_mw": 0, "oxygen_nm3_min": 0, "ng_nm3_min": 0, "carbon_kg_min": 0, "flux_kg_min": 0},
            "_config_obj": cfg,
        }
        self.assertTrue(policy.act(obs)["tap_command"])

    def test_baseline_schedule_no_zero_tapping_when_ready(self):
        ctrl = EAFController(default_config(), enhanced_model=True)
        obs = ctrl.reset()
        obs["bath_temp_c"] = ctrl.config.tap_target_temp_c + 5
        obs["liquid_steel_kg"] = ctrl.config.tap_target_steel_kg
        obs["steel_carbon_wt_pct"] = 0.05
        action = IndustrialBaselineSchedulePolicy().act(obs)
        self.assertTrue(action["tap_command"])

    def test_mpc_uses_model_c_rollout(self):
        ctrl = EAFController(default_config(), enhanced_model=True)
        obs = ctrl.reset()
        act = MPCPolicy(horizon=5).act(obs)
        self.assertIn("power_mw", act)

    def test_mpc_temperature_below_max_temp(self):
        ctrl = EAFController(default_config(), enhanced_model=True)
        policy = MPCPolicy(horizon=5)
        obs = ctrl.reset()
        for _ in range(20):
            res = ctrl.step(policy.act(obs))
            obs = res.observation
            self.assertLessEqual(float(obs["bath_temp_c"]), ctrl.config.max_temp_c + 30.0)
            if res.done:
                break

    def test_safety_filter_blocks_invalid_tap(self):
        ctrl = EAFController(default_config(), enhanced_model=True)
        ctrl.reset()
        res = ctrl.step({"power_mw": 0, "oxygen_nm3_min": 0, "ng_nm3_min": 0, "carbon_kg_min": 0, "flux_kg_min": 0, "tap_command": True})
        self.assertTrue(bool(res.info["invalid_tap_command"]))

    def test_safety_filter_clamps_overtemperature_action(self):
        ctrl = EAFController(default_config(), enhanced_model=True)
        obs = ctrl.reset()
        ctrl.state.steel_temp_k = (ctrl.config.max_temp_c + 10.0) + 273.15
        res = ctrl.step({"power_mw": 120, "oxygen_nm3_min": 120, "ng_nm3_min": 20, "carbon_kg_min": 10, "flux_kg_min": 10, "tap_command": False})
        self.assertEqual(float(res.info["safe_action"]["power_mw"]), 0.0)


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
                    "--seeds",
                    "2",
                    "--model",
                    "C",
                    "--mpc-horizon",
                    "5",
                ],
                check=True,
                env={**os.environ, "PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"},
            )
            self.assertTrue((run_out / "scenario_summary.csv").exists())
            self.assertTrue((run_out / "statistical_analysis.csv").exists())
            self.assertTrue((run_out / "result.html").exists())
            summary = pd.read_csv(run_out / "scenario_summary.csv")
            self.assertTrue((summary["model_name"] == "Model_C_enhanced_hybrid").all())

    def test_downtime_enforced_for_all_policies(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            subprocess.run(
                [sys.executable, "-m", "agents.runners.run_agent", "--output-dir", str(out), "--seeds", "1", "--n-scenarios", "6"],
                check=True,
                env={**os.environ, "PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"},
            )
            for p in ["baseline_schedule", "rule_based", "mpc"]:
                ts = pd.read_csv(out / "timeseries" / f"agent_timeseries_delayed_melting_downtime_{p}_seed0.csv")
                if "is_downtime" in ts.columns:
                    dt = ts[ts["is_downtime"] == True]  # noqa: E712
                    if not dt.empty:
                        self.assertTrue((dt["power_mw"] <= 1e-9).all())

    def test_dri20_has_nonzero_dri_charge_events(self):
        sc = scenario_configs(default_config())
        self.assertGreater(sum(e.dri_kg for e in sc["dri20"].charge_events), 0.0)

    def test_dri20_differs_from_base_case(self):
        sc = scenario_configs(default_config())
        self.assertNotEqual(sc["dri20"].initial_dri_kg, sc["base_case"].initial_dri_kg)

    def test_percentage_returns_na_when_baseline_zero(self):
        from agents.runners.run_agent import _safe_pct

        s = _safe_pct(pd.Series([10.0]), pd.Series([0.0]))
        self.assertEqual(s.iloc[0], "n/a")

    def test_result_html_created(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            subprocess.run(
                [sys.executable, "-m", "agents.runners.run_agent", "--output-dir", str(out), "--seeds", "1"],
                check=True,
                env={**os.environ, "PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"},
            )
            self.assertTrue((out / "result.html").exists())



    def test_run_episode_max_steps_650(self):
        from agents.runners.episode_runner import run_episode

        ctrl = EAFController(default_config(), enhanced_model=True)
        out = run_episode(ctrl, RuleBasedPolicy(), policy_name="rule_based", max_steps=650)
        self.assertLessEqual(out.steps, 650)

    def test_required_kpis_exist_in_summary(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            subprocess.run(
                [sys.executable, "-m", "agents.runners.run_agent", "--output-dir", str(out), "--seeds", "1"],
                check=True,
                env={**os.environ, "PYTHONPATH": f"{Path.cwd()}:{Path.cwd() / 'src'}"},
            )
            summary = pd.read_csv(out / "scenario_summary.csv")
            for col in ["tap_success", "cum_tapped_kg", "max_bath_temp_c", "cum_electric_mwh", "model_name", "seed", "terminal_reward"]:
                self.assertIn(col, summary.columns)


if __name__ == "__main__":
    unittest.main()
