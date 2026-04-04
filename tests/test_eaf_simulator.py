"""Compatibility smoke test for legacy entry script."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestLegacyWrapper(unittest.TestCase):
    def test_legacy_script_runs(self):
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, "eaf_simulator.py", "--config", "configs/base_case.json", "--output-dir", td]
            cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(cp.returncode, 0, msg=cp.stderr)
            self.assertTrue((Path(td) / "summary_all_scenarios.csv").exists())


class TestLegacyWrapper(unittest.TestCase):
    """Compatibility smoke test for script entrypoint."""

    def test_legacy_script_runs(self) -> None:
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as td:
            cp = subprocess.run([sys.executable, "eaf_simulator.py", "--output-dir", td], capture_output=True, text=True, check=False)
            self.assertEqual(cp.returncode, 0, msg=cp.stderr)
            self.assertTrue((Path(td) / "summary_all_scenarios.csv").exists())
            self.assertTrue((Path(td) / "result.html").exists())


if __name__ == "__main__":
    unittest.main()
