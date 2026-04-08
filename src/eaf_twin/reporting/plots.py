from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_core(df: pd.DataFrame, out_dir: Path, scenario: str, model_name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = df["time_min"]
    paths = []

    # Detect dynamic events to mark explicitly on the plots
    charge_times = t[df["solid_scrap_kg"].diff() > 1000]
    
    tap_mask = df["tapped_kg_s"] > 0
    tap_start, tap_end = None, None
    if tap_mask.any():
        tap_start = t[tap_mask].iloc[0]
        tap_end = t[tap_mask].iloc[-1]

    def format_plot(ax, title, ylabel):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time [min]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({scenario} | {model_name})")
        
        # Draw explicit operational events
        for ct in charge_times:
            ax.axvline(x=ct, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
        if tap_start is not None and tap_end is not None:
            ax.axvspan(tap_start, tap_end, color="red", alpha=0.1, label="Tapping Phase")

    def save(metric_name: str) -> Path:
        path = out_dir / f"plot_{scenario}_{metric_name}_{model_name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return path

    # 1. Temperature trajectories
    fig, ax = plt.subplots(figsize=(9, 5))
    t_mm = df["t_mm_c"] if "t_mm_c" in df.columns else df["liquid_steel_temp_c"]
    t_ss = df["t_ss_c"] if "t_ss_c" in df.columns else df["solid_scrap_temp_c"]

    ax.plot(t, t_mm, label="Liquid & Slag Bath ($T_{mm}$)", color="blue", linewidth=1.8)
    ax.plot(t, t_ss, label="Solid Scrap Bulk ($T_{ss}$)", color="red", linestyle="--", linewidth=1.8)
    ax.plot(t, df["offgas_temp_c"], label="Off-gas Temp", color="green", linewidth=1.2)
    
    format_plot(ax, "Temperature trajectories", "Temperature [°C]")
    ax.legend(loc="lower right")
    paths.append(save("temperatures"))

    # 2. Melted fraction
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, df["melted_fraction"], color="tab:purple", linewidth=2)
    format_plot(ax, "Melted fraction", "Fraction")
    paths.append(save("melted_fraction"))

    # 3. Metal phase masses
    fig, ax = plt.subplots(figsize=(9, 5))
    solid_metal = df["solid_scrap_kg"] + df.get("solid_dri_kg", 0.0)
    ax.plot(t, solid_metal, label="Remaining Solid Scrap", color="tab:blue", linewidth=1.8)
    ax.plot(t, df["liquid_steel_kg"], label="Liquid Steel Bath", color="tab:orange", linewidth=1.8)
    
    
    format_plot(ax, "Metal phase masses", "Mass [kg]")
    ax.legend(loc="center right")
    paths.append(save("metal_masses"))

    # 4. Cumulative energies
    fig, ax = plt.subplots(figsize=(9, 5))
    cum_chemical_mwh = df["cum_chemical_gj"] / 3.6
    ax.plot(t, df["cum_electric_mwh"], label="Electric [MWh]", linewidth=1.8)
    ax.plot(t, cum_chemical_mwh, label="Chemical [MWh eq]", linewidth=1.8)
    format_plot(ax, "Cumulative energies", "Energy [MWh]")
    ax.legend()
    paths.append(save("cumulative_energy"))

    # 5. Cumulative consumables
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, df["cum_oxygen_nm3"], label="Oxygen [Nm3]", linewidth=1.8)
    ax.plot(t, df["cum_ng_nm3"], label="Natural gas [Nm3]", linewidth=1.8)
    ax.plot(t, df["cum_carbon_kg"], label="Carbon [kg]", linewidth=1.8)
    format_plot(ax, "Cumulative consumables", "Consumption")
    ax.legend()
    paths.append(save("consumables"))

    # 6. Carbon trajectory
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, df["steel_carbon_wt_pct"], color="tab:brown", linewidth=1.8)
    format_plot(ax, "Carbon trajectory", "Steel carbon [wt%]")
    paths.append(save("steel_carbon"))

    # 7. Heat flow stack
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(
        t,
        df["q_useful_mw"],
        df["q_melt_mw"],
        df["q_loss_mw"],
        labels=["Useful Input (Arc+Chem)", "Melting Sink", "Losses (Wall+Rad)"],
        alpha=0.85,
    )
    format_plot(ax, "Heat flow stack", "Power [MW]")
    ax.legend(loc="upper right")
    paths.append(save("heat_stack"))

    return paths
