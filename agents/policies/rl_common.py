from __future__ import annotations

from dataclasses import dataclass

from agents.types import ActionDict, ObservationDict


DISCRETE_ACTIONS: dict[str, ActionDict] = {
    "low_power": {"power_mw": 45.0, "oxygen_nm3_min": 28.0, "ng_nm3_min": 6.0, "carbon_kg_min": 4.0, "flux_kg_min": 30.0, "tap_command": False},
    "medium_power": {"power_mw": 78.0, "oxygen_nm3_min": 58.0, "ng_nm3_min": 14.0, "carbon_kg_min": 10.0, "flux_kg_min": 95.0, "tap_command": False},
    "high_power": {"power_mw": 105.0, "oxygen_nm3_min": 84.0, "ng_nm3_min": 22.0, "carbon_kg_min": 16.0, "flux_kg_min": 145.0, "tap_command": False},
    "refining": {"power_mw": 58.0, "oxygen_nm3_min": 65.0, "ng_nm3_min": 9.0, "carbon_kg_min": 8.0, "flux_kg_min": 70.0, "tap_command": False},
    "holding": {"power_mw": 20.0, "oxygen_nm3_min": 10.0, "ng_nm3_min": 3.0, "carbon_kg_min": 2.0, "flux_kg_min": 15.0, "tap_command": False},
    "tap_if_ready": {"power_mw": 12.0, "oxygen_nm3_min": 3.0, "ng_nm3_min": 1.0, "carbon_kg_min": 0.0, "flux_kg_min": 0.0, "tap_command": True},
}

ACTION_NAMES = list(DISCRETE_ACTIONS.keys())


@dataclass
class Discretizer:
    time_bin_min: float = 5.0
    temp_bin_c: float = 40.0
    melt_bin: float = 0.1
    carbon_bin: float = 0.01

    def encode(self, obs: ObservationDict) -> str:
        return "|".join(
            [
                f"t:{int(float(obs.get('time_min', 0.0)) / self.time_bin_min)}",
                f"temp:{int(float(obs.get('bath_temp_c', 0.0)) / self.temp_bin_c)}",
                f"melt:{int(float(obs.get('melted_fraction', 0.0)) / self.melt_bin)}",
                f"c:{int(float(obs.get('steel_carbon_wt_pct', 0.05)) / self.carbon_bin)}",
                f"tap:{int(bool(obs.get('can_tap', False)))}",
                f"down:{int(bool(obs.get('is_downtime', False)))}",
            ]
        )


def safe_discrete_action(name: str, obs: ObservationDict) -> ActionDict:
    action = dict(DISCRETE_ACTIONS.get(name, DISCRETE_ACTIONS["medium_power"]))
    if name == "tap_if_ready" and not bool(obs.get("can_tap", False)):
        action["tap_command"] = False
    return action


def normalized_obs_vec(obs: ObservationDict) -> list[float]:
    return [
        min(1.0, float(obs.get("time_min", 0.0)) / 70.0),
        min(1.2, max(0.0, (float(obs.get("bath_temp_c", 0.0)) - 1200.0) / 700.0)),
        min(1.2, max(0.0, (float(obs.get("scrap_temp_c", 0.0)) - 20.0) / 1700.0)),
        min(1.2, max(0.0, (float(obs.get("slag_temp_c", 0.0)) - 20.0) / 1700.0)),
        min(1.0, max(0.0, float(obs.get("melted_fraction", 0.0)))),
        min(1.0, float(obs.get("remaining_solid_kg", 0.0)) / 120000.0),
        min(1.2, float(obs.get("liquid_steel_kg", 0.0)) / 120000.0),
        min(1.0, float(obs.get("steel_carbon_wt_pct", 0.05)) / 0.2),
        min(1.5, float(obs.get("cum_electric_mwh", 0.0)) / 120.0),
        min(1.5, float(obs.get("cum_oxygen_nm3", 0.0)) / 7000.0),
        min(1.5, float(obs.get("cum_ng_nm3", 0.0)) / 2800.0),
        float(bool(obs.get("can_tap", False))),
        float(bool(obs.get("is_downtime", False))),
    ]
