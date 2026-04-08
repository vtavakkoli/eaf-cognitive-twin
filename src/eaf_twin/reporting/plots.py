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

    # Temperature trajectories (Corrected with Bulk mass-weighted T_mm and T_ss formulas)
    plt.figure(figsize=(9, 5))
    
    # Check if the model explicitly provided the bulk calculations, else compute them
    if "t_mm_c" in df.columns and "t_ss_c" in df.columns:
        t_mm = df["t_mm_c"]
        t_ss = df["t_ss_c"]
    else:
        m_slag = df["slag_kg"]
        t_slag = df["slag_temp_c"]
        m_liq = df["liquid_steel_kg"]
        t_liq = df["liquid_steel_temp_c"]
        
        # Tmm: Mass-weighted average of the already melted material (Slag + Liquid Steel)
        t_mm = (m_slag * t_slag + m_liq * t_liq) / (m_liq + m_slag).clip(lower=1e-9)
        # Tss: Solid material temperature (dominated by solid scrap)
        t_ss = df["solid_scrap_temp_c"]

    # Plot the aggregated T_mm and T_ss
    plt.plot(t, t_mm, label="Already melted material ($T_{mm}$)", color="blue", linewidth=1.5)
    plt.plot(t, t_ss, label="Solid material ($T_{ss}$)", color="red", linestyle="--", linewidth=1.5)
    
    # Plot the individual component temperatures for context (slightly transparent)
    plt.plot(t, df["liquid_steel_temp_c"], label="Liquid steel component", color="cyan", linewidth=1.0, alpha=0.5)
    plt.plot(t, df["slag_temp_c"], label="Slag component", color="orange", linewidth=1.0, alpha=0.6)
    plt.plot(t, df["offgas_temp_c"], label="Off-gas", color="green", linewidth=1.5)
    
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

    # Remaining solid metallic charge
    plt.figure(figsize=(9, 5))
    plt.plot(t, df.get("remaining_solid_kg", df["solid_scrap_kg"] + df.get("solid_dri_kg", 0.0)), color="tab:red")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [min]")
    plt.ylabel("Remaining solid [kg]")
    plt.title(f"Remaining solid metallic charge ({scenario} | {model_name})")
    paths.append(save("remaining_solid"))

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
