"""CLI smoke test for report generation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCliRun(unittest.TestCase):
    def test_cli_generates_summary_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{env.get('PYTHONPATH', '')}"
            cp = subprocess.run(
                [sys.executable, "-m", "eaf_twin.cli", "run", "--config", "configs/base_case.json", "--output-dir", td],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(cp.returncode, 0, msg=cp.stderr)
            self.assertTrue((Path(td) / "summary_all_scenarios.csv").exists())
            self.assertTrue((Path(td) / "result.html").exists())


if __name__ == "__main__":
    unittest.main()
