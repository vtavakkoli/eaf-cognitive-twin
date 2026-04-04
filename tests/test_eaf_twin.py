from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eaf_twin.config.defaults import default_config, scenario_configs
from eaf_twin.config.loader import ConfigError, load_config, validate_config
from eaf_twin.domain.models import ChargeEvent, StageWindow
from eaf_twin.io.persistence import save_summary_table, save_time_series
from eaf_twin.models.first_principles import FirstPrinciplesModel
from eaf_twin.simulation.runner import run_full_simulation
from eaf_twin.simulation.schedule import active_setpoints


class TestConfigValidation(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = default_config()
        validate_config(cfg)

    def test_invalid_stage_raises(self):
        cfg = default_config()
        cfg.stage_windows = [StageWindow(10, 5, 1, 1, 1, 1, 1)]
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_load_unknown_key_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text('{"unknown": 1}')
            with self.assertRaises(ConfigError):
                load_config(p)


class TestScheduleAndCharge(unittest.TestCase):
    def test_stage_schedule_selection(self):
        cfg = default_config()
        cfg.downtime_start_min = 12
        cfg.downtime_duration_min = 2
        normal = active_setpoints(cfg, 2)
        down = active_setpoints(cfg, 12.5)
        self.assertGreater(normal["power_mw"], 0)
        self.assertEqual(down["power_mw"], 0)

    def test_charge_event_applied(self):
        cfg = default_config()
        cfg.charge_events = [ChargeEvent(1.0, scrap_kg=1000, dri_kg=500)]
        model = FirstPrinciplesModel(cfg, enhanced=False)
        st = model.initialize_state()
        model.apply_charge_events(st, 0, 61)
        self.assertGreaterEqual(st.solid_dri_kg, 500)


class TestPhysicsAndKPIs(unittest.TestCase):
    def _short_cfg(self):
        cfg = default_config()
        cfg.heat_duration_min = 8
        cfg.dt_s = 2
        return cfg

    def test_tapping_logic_and_cumulative_tapped(self):
        cfg = self._short_cfg()
        res = FirstPrinciplesModel(cfg, enhanced=True).simulate()
        self.assertIn("cum_tapped_kg", res.df.columns)
        self.assertGreaterEqual(float(res.df["cum_tapped_kg"].iloc[-1]), 0.0)

    def test_useful_heat_accumulates(self):
        cfg = self._short_cfg()
        res = FirstPrinciplesModel(cfg, enhanced=True).simulate()
        series = res.df["cum_useful_heat_gj"]
        self.assertTrue((series.diff().fillna(0) >= -1e-9).all())

    def test_no_negative_masses(self):
        cfg = self._short_cfg()
        res = FirstPrinciplesModel(cfg, enhanced=True).simulate()
        cols = ["solid_scrap_kg", "solid_dri_kg", "liquid_steel_kg", "slag_kg"]
        self.assertTrue((res.df[cols] >= -1e-9).all().all())

    def test_no_melting_when_below_melt_condition(self):
        cfg = self._short_cfg()
        cfg.heat_duration_min = 2
        cfg.initial_hot_heel_kg = 0.0
        cfg.initial_steel_temp_c = 50.0
        cfg.initial_scrap_kg = 20_000.0
        cfg.charge_events = [ChargeEvent(0.0, scrap_kg=20_000.0, dri_kg=0.0)]
        cfg.stage_windows = [StageWindow(0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
        res = FirstPrinciplesModel(cfg, enhanced=False).simulate()
        self.assertTrue((res.df["phase_region"] == "solid_heating").all())
        self.assertTrue((res.df["melt_rate_kg_s"] == 0.0).all())
        self.assertAlmostEqual(float(res.df["solid_scrap_kg"].iloc[-1]), float(res.df["solid_scrap_kg"].iloc[0]), places=6)

    def test_phase_and_superheat_are_state_conditional(self):
        cfg = self._short_cfg()
        cfg.heat_duration_min = 6
        cfg.initial_hot_heel_kg = 30_000.0
        cfg.initial_steel_temp_c = cfg.steel_melt_temp_c
        cfg.initial_scrap_kg = 3_000.0
        cfg.charge_events = [ChargeEvent(0.0, scrap_kg=3_000.0, dri_kg=0.0)]
        cfg.stage_windows = [StageWindow(0.0, 6.0, 200.0, 0.0, 0.0, 0.0, 0.0)]
        res = FirstPrinciplesModel(cfg, enhanced=False).simulate()
        phase_rows = res.df[res.df["phase_region"] == "phase_change"]
        self.assertGreater(len(phase_rows), 0)
        self.assertTrue((phase_rows["melt_rate_kg_s"] > 0).any())
        superheat_rows = res.df[res.df["phase_region"] == "liquid_superheat"]
        self.assertGreater(len(superheat_rows), 0)
        self.assertTrue((superheat_rows["solid_scrap_kg"] <= 1e-6).all())

    def test_near_fully_melted_requires_hot_liquid_bath(self):
        cfg = default_config()
        cfg.heat_duration_min = 65
        res = FirstPrinciplesModel(cfg, enhanced=True).simulate().df
        near_full = res[res["melted_fraction"] > 0.98]
        if not near_full.empty:
            self.assertTrue((near_full["liquid_steel_temp_c"] >= cfg.steel_melt_temp_c - 10.0).all())

    def test_charge_event_drops_solid_temp_more_than_liquid_temp(self):
        cfg = default_config()
        cfg.heat_duration_min = 20
        res = FirstPrinciplesModel(cfg, enhanced=False).simulate().df.reset_index(drop=True)
        charge_time = min(ev.time_min for ev in cfg.charge_events if ev.time_min > 0)
        before_idx = int(res.index[res["time_min"] < charge_time][-1])
        after_idx = int(res.index[res["time_min"] >= charge_time][0])
        solid_drop = float(res.loc[after_idx, "solid_scrap_temp_c"] - res.loc[before_idx, "solid_scrap_temp_c"])
        liquid_drop = float(res.loc[after_idx, "liquid_steel_temp_c"] - res.loc[before_idx, "liquid_steel_temp_c"])
        self.assertLess(solid_drop, -20.0)
        self.assertGreater(liquid_drop, solid_drop)


class TestScenarioAndOutputs(unittest.TestCase):
    def test_scenario_generation(self):
        cfg = default_config()
        scenarios = scenario_configs(cfg)
        self.assertIn("dri20", scenarios)

    def test_output_generation(self):
        cfg = default_config()
        cfg.heat_duration_min = 4
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            df = run_full_simulation(cfg, out)
            self.assertFalse(df.empty)
            self.assertTrue((out / "summary_all_scenarios.csv").exists())
            ts = out / f"timeseries_base_case_Model_A_empirical.csv"
            self.assertTrue(ts.exists())
            save_time_series(df, out, "tmp.csv")
            save_summary_table(df.to_dict(orient="records"), out, "tmp")


if __name__ == "__main__":
    unittest.main()
