from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_core(df: pd.DataFrame, out_dir: Path, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = df["time_min"]
    paths = []

    def save(name: str) -> Path:
        path = out_dir / f"plot_{prefix}_{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return path

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["steel_temp_c"], label="steel")
    plt.plot(t, df["slag_temp_c"], label="slag")
    plt.plot(t, df["offgas_temp_c"], label="offgas")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xlabel("Time [min]"); plt.ylabel("Temperature [C]")
    paths.append(save("temperatures"))

    plt.figure(figsize=(9, 5))
    plt.plot(t, df["cum_tapped_kg"], label="cum tapped")
    plt.grid(True, alpha=0.3); plt.legend(); plt.xlabel("Time [min]"); plt.ylabel("kg")
    paths.append(save("tapping"))
    return paths
