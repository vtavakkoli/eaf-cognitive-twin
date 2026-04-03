from __future__ import annotations

from eaf_twin.constants import SECONDS_PER_MIN, SIGMA
from eaf_twin.models.base import BaseEAFModel, start_or_continue_tapping
from eaf_twin.simulation.schedule import stage_name
from eaf_twin.units import clamp


class EmpiricalModel(BaseEAFModel):
    name = "Model_A_empirical"

    def simulate(self):
        cfg = self.config

        def step(state, inputs, warnings):
            dt = cfg.dt_s
            stg = stage_name(state.time_s / SECONDS_PER_MIN, state.melted_fraction)
            power_w = inputs["power_mw"] * 1e6
            q_elec = power_w * dt
            q_burn = cfg.eta_burner * inputs["ng_nm3_min"] / 60.0 * cfg.lhv_ng_j_nm3 * dt
            q_oxy = 0.62 * cfg.oxygen_reaction_efficiency * inputs["oxygen_nm3_min"] / 60.0 * cfg.oxygen_heat_j_nm3 * dt
            eta = {"bore_in": 0.58, "main_melting": 0.72, "refining": 0.63, "superheat": 0.60, "tapping": 0.40}[stg]
            useful = eta * q_elec + 0.75 * q_burn + q_oxy

            t_k = (state.steel_temp_c + state.slag_temp_c) * 0.5 + 273.15
            amb_k = cfg.ambient_temp_c + 273.15
            q_wall = cfg.ua_wall_w_k * (t_k - amb_k) * dt
            q_rad = cfg.radiation_loss_factor * SIGMA * cfg.area_effective_m2 * (t_k**4 - amb_k**4) * dt
            q_loss = max(0.0, q_wall + max(0.0, q_rad))

            q_need_scrap = cfg.cp_scrap_j_kgk * (cfg.steel_melt_temp_c - cfg.scrap_temp_c) + cfg.latent_heat_steel_j_kg
            q_need_dri = cfg.cp_dri_j_kgk * (cfg.steel_melt_temp_c - cfg.scrap_temp_c) + cfg.latent_heat_steel_j_kg + cfg.dri_reduction_endotherm_j_kg
            q_avail = max(0.0, useful - q_loss)
            melt_scrap = min(state.solid_scrap_kg, q_avail / max(q_need_scrap, 1e-9))
            q_left = q_avail - melt_scrap * q_need_scrap
            melt_dri = min(state.solid_dri_kg, max(0.0, q_left) / max(q_need_dri, 1e-9))
            q_melt = melt_scrap * q_need_scrap + melt_dri * q_need_dri
            state.solid_scrap_kg -= melt_scrap
            state.solid_dri_kg -= melt_dri
            state.liquid_steel_kg += melt_scrap + melt_dri * cfg.dri_fe_metallization
            state.feo_slag_kg += melt_dri * (1.0 - cfg.dri_fe_metallization)

            heat_cap = max(cfg.cp_steel_j_kgk * max(state.liquid_steel_kg, 10_000.0), 1e-9)
            d_t = (q_avail - q_melt) / heat_cap
            state.steel_temp_c += d_t
            state.slag_temp_c += 0.35 * d_t
            state.offgas_temp_c = clamp(cfg.ambient_temp_c + 0.1 * (state.steel_temp_c - cfg.ambient_temp_c), cfg.ambient_temp_c, cfg.max_offgas_temp_c)
            state.slag_kg += inputs["flux_kg_min"] / 60.0 * dt * cfg.flux_to_slag_factor
            decarb = min(state.steel_carbon_kg, inputs["oxygen_nm3_min"] / 60.0 * dt * cfg.decarb_kg_per_nm3_o2 * 0.7)
            inj_c = inputs["carbon_kg_min"] / 60.0 * dt
            state.steel_carbon_kg += inj_c - decarb
            tapped = start_or_continue_tapping(state, cfg)

            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_burn + q_oxy
            state.cum_useful_heat_j += useful
            state.cum_losses_j += q_loss
            state.cum_oxygen_nm3 += inputs["oxygen_nm3_min"] / 60.0 * dt
            state.cum_ng_nm3 += inputs["ng_nm3_min"] / 60.0 * dt
            state.cum_carbon_kg += inj_c
            return {"stage": stg, "q_useful_mw": useful / dt / 1e6, "q_melt_mw": q_melt / dt / 1e6, "q_loss_mw": q_loss / dt / 1e6, "tapped_kg_s": tapped / dt}

        return self.run_loop(step)
