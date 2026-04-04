from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_core(df: pd.DataFrame, out_dir: Path, scenario: str, model_name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = df["time_min"]
    paths = []

    def save(metric_name: str) -> Path:
        path = out_dir / f"plot_{scenario}_{metric_name}_{model_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return path

    # Temperature trajectories
    plt.figure(figsize=(9, 5))
    plt.plot(t, df["liquid_steel_temp_c"], label="Liquid steel")
    plt.plot(t, df["slag_temp_c"], label="Slag")
    plt.plot(t, df["offgas_temp_c"], label="Off-gas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [min]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Temperature trajectories ({scenario} | {model_name})")
    paths.append(save("temperatures"))

    # Melted fraction
    plt.figure(figsize=(9, 5))
    plt.plot(t, df["melted_fraction"], color="tab:purple")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [min]")
    plt.ylabel("Melted fraction")
    plt.title(f"Melted fraction ({scenario} | {model_name})")
    paths.append(save("melted_fraction"))

    # Metal phase masses
    solid_metal = df["solid_scrap_kg"] + df.get("solid_dri_kg", 0.0)
    plt.figure(figsize=(9, 5))
    plt.plot(t, solid_metal, label="Solid scrap")
    plt.plot(t, df["liquid_steel_kg"], label="Liquid steel")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("Time [min]")
    plt.ylabel("Mass [kg]")
    plt.title(f"Metal phase masses ({scenario} | {model_name})")
    paths.append(save("metal_masses"))

    # Cumulative energies
    cum_chemical_mwh = df["cum_chemical_gj"] / 3.6
    plt.figure(figsize=(9, 5))
    plt.plot(t, df["cum_electric_mwh"], label="Electric [MWh]")
    plt.plot(t, cum_chemical_mwh, label="Chemical [MWh eq]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("Time [min]")
    plt.ylabel("Cumulative energy [MWh]")
    plt.title(f"Cumulative energies ({scenario} | {model_name})")
    paths.append(save("cumulative_energy"))

    # Cumulative consumables
    plt.figure(figsize=(9, 5))
    plt.plot(t, df["cum_oxygen_nm3"], label="Oxygen [Nm3]")
    plt.plot(t, df["cum_ng_nm3"], label="Natural gas [Nm3]")
    plt.plot(t, df["cum_carbon_kg"], label="Carbon [kg]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("Time [min]")
    plt.ylabel("Cumulative consumption")
    plt.title(f"Cumulative consumables ({scenario} | {model_name})")
    paths.append(save("consumables"))

    # Carbon trajectory
    plt.figure(figsize=(9, 5))
    plt.plot(t, df["steel_carbon_wt_pct"], color="tab:brown")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [min]")
    plt.ylabel("Steel carbon [wt%]")
    plt.title(f"Carbon trajectory ({scenario} | {model_name})")
    paths.append(save("steel_carbon"))

    # Heat flow stack
    plt.figure(figsize=(9, 5))
    plt.stackplot(
        t,
        df["q_useful_mw"],
        df["q_melt_mw"],
        df["q_loss_mw"],
        labels=["Useful in", "Melting sink", "Losses"],
        alpha=0.85,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.xlabel("Time [min]")
    plt.ylabel("Power [MW]")
    plt.title(f"Heat flow stack ({scenario} | {model_name})")
    paths.append(save("heat_stack"))

    return paths
