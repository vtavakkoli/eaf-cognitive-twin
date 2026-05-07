from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_time_series(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def save_summary_table(rows: list[dict], out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"summary_{name}.csv"
    json_path = out_dir / f"summary_{name}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    return csv_path, json_path
