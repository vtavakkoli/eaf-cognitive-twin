"""Unit tests for eaf_simulator.py."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PANDAS = importlib.util.find_spec("pandas") is not None
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAS_CORE_DEPS = HAS_NUMPY and HAS_PANDAS and HAS_MATPLOTLIB


@unittest.skipUnless(HAS_CORE_DEPS, "Core dependencies missing: numpy/pandas/matplotlib")
class TestEAFConfigAndSchedule(unittest.TestCase):
    """Test configuration and schedule helper behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        import eaf_simulator as sim

        cls.sim = sim

    def test_default_config_and_scenarios(self) -> None:
        cfg = self.sim.load_config(None)
        self.assertEqual(len(cfg.stage_windows), 5)
        self.assertGreaterEqual(len(cfg.charge_events), 1)

        scenarios = self.sim.scenario_configs(cfg)
        required = {
            "base_case",
            "higher_oxygen",
            "higher_natural_gas",
            "improved_foamy_slag",
            "dri20",
            "delayed_melting_downtime",
        }
        self.assertTrue(required.issubset(scenarios.keys()))

    def test_active_setpoints_with_downtime(self) -> None:
        cfg = self.sim.load_config(None)
        cfg.downtime_start_min = 12.0
        cfg.downtime_duration_min = 4.0

        normal = self.sim.active_setpoints(cfg, 2.0)
        down = self.sim.active_setpoints(cfg, 13.0)

        self.assertGreater(normal["power_mw"], 0.0)
        self.assertEqual(down["power_mw"], 0.0)
        self.assertEqual(down["carbon_kg_min"], 0.0)


@unittest.skipUnless(HAS_CORE_DEPS, "Core dependencies missing: numpy/pandas/matplotlib")
class TestEAFModels(unittest.TestCase):
    """Test each model fidelity with a short deterministic run."""

    @classmethod
    def setUpClass(cls) -> None:
        import eaf_simulator as sim

        cls.sim = sim

    def _short_config(self):
        cfg = self.sim.load_config(None)
        cfg.heat_duration_min = 6.0
        cfg.dt_s = 2.0
        cfg.measurement_noise_std = 0.0
        return cfg

    def test_model_a_runs(self) -> None:
        cfg = self._short_config()
        res = self.sim.EmpiricalModel(cfg).simulate()
        self.assertFalse(res.df.empty)
        self.assertIn("steel_temp_c", res.df.columns)
        self.assertGreaterEqual(res.summary["heat_time_min"], 1.0)

    def test_model_b_runs(self) -> None:
        cfg = self._short_config()
        res = self.sim.FirstPrinciplesModel(cfg, enhanced=False).simulate()
        self.assertFalse(res.df.empty)
        self.assertIn("offgas_co_frac", res.df.columns)
        self.assertGreaterEqual(res.summary["tap_steel_kg"], 0.0)

    def test_model_c_runs(self) -> None:
        cfg = self._short_config()
        cfg.initial_dri_fraction = 0.2
        res = self.sim.FirstPrinciplesModel(cfg, enhanced=True).simulate()
        self.assertFalse(res.df.empty)
        self.assertIn("foamy_factor", res.df.columns)
        self.assertGreaterEqual(res.summary["thermal_efficiency_est"], 0.0)


@unittest.skipUnless(HAS_CORE_DEPS, "Core dependencies missing: numpy/pandas/matplotlib")
class TestOutputsAndSensitivity(unittest.TestCase):
    """Test output writing and sensitivity module."""

    @classmethod
    def setUpClass(cls) -> None:
        import eaf_simulator as sim

        cls.sim = sim

    def test_time_series_summary_and_sensitivity_files(self) -> None:
        cfg = self.sim.load_config(None)
        cfg.heat_duration_min = 5.0
        cfg.dt_s = 2.0

        result = self.sim.FirstPrinciplesModel(cfg, enhanced=True).simulate()

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            p_ts = self.sim.save_time_series(result, out_dir)
            self.assertTrue(p_ts.exists())

            rows = [{"scenario": "x", "model": result.model_name, **result.summary, "status": "ok", "issues": ""}]
            p_csv, p_json = self.sim.save_summary_table(rows, out_dir, "unit")
            self.assertTrue(p_csv.exists())
            self.assertTrue(p_json.exists())

            sens_df, sens_plot = self.sim.run_sensitivity(cfg, out_dir)
            self.assertFalse(sens_df.empty)
            self.assertTrue(sens_plot.exists())


if __name__ == "__main__":
    unittest.main()
