from __future__ import annotations

import math

from eaf_twin.constants import SECONDS_PER_MIN, SIGMA
from eaf_twin.models.base import BaseEAFModel, start_or_continue_tapping
from eaf_twin.simulation.schedule import smooth_step, stage_name
from eaf_twin.units import clamp


class FirstPrinciplesModel(BaseEAFModel):
    name = "Model_B_first_principles"

    def __init__(self, config, enhanced: bool = False):
        super().__init__(config)
        self.enhanced = enhanced
        if enhanced:
            self.name = "Model_C_enhanced_hybrid"

    def simulate(self):
        cfg = self.config

        def step(state, inputs, warnings):
            dt = cfg.dt_s
            stg = stage_name(state.time_s / SECONDS_PER_MIN, state.melted_fraction)
            power_w = inputs["power_mw"] * 1e6
            o2_flow = inputs["oxygen_nm3_min"] / 60.0
            ng_flow = inputs["ng_nm3_min"] / 60.0
            c_flow = inputs["carbon_kg_min"] / 60.0
            flux_flow = inputs["flux_kg_min"] / 60.0

            foam = 0.35
            if self.enhanced:
                ratio = c_flow / max(o2_flow * 0.18, 1e-3)
                foam = clamp(0.35 + 0.22 * math.tanh(1.6 * (ratio - 1.0)), 0.05, 0.95)
            eta = {"bore_in": cfg.eta_arc_bore_in, "main_melting": cfg.eta_arc_melting, "refining": cfg.eta_arc_refining, "superheat": cfg.eta_arc_superheat, "tapping": 0.4}[stg]
            eta += 0.04 * smooth_step(state.melted_fraction, 0.2, 0.95)
            eta += 0.05 * foam
            eta = clamp(eta, 0.4, 0.92)

            q_elec = power_w * dt
            q_arc_useful = eta * q_elec
            q_burn = cfg.eta_burner * ng_flow * cfg.lhv_ng_j_nm3 * dt
            q_oxy = o2_flow * cfg.oxygen_heat_j_nm3 * cfg.oxygen_reaction_efficiency * dt
            q_c = c_flow * cfg.carbon_heat_j_kg * cfg.carbon_reaction_efficiency * dt
            q_chem = q_burn + q_oxy + q_c

            fe_oxid = min(state.liquid_steel_kg * 0.0015, o2_flow * cfg.fe_oxidation_ratio_per_nm3_o2 * dt)
            oxide = 1.29 * fe_oxid
            decarb = min(state.steel_carbon_kg, o2_flow * cfg.decarb_kg_per_nm3_o2 * dt)
            state.steel_carbon_kg += c_flow * dt - decarb - 0.0006 * state.steel_carbon_kg * dt
            state.feo_slag_kg += oxide
            state.liquid_steel_kg -= fe_oxid

            q_need_scrap = cfg.cp_scrap_j_kgk * (cfg.steel_melt_temp_c - cfg.scrap_temp_c) + cfg.latent_heat_steel_j_kg
            q_need_dri = cfg.cp_dri_j_kgk * (cfg.steel_melt_temp_c - cfg.scrap_temp_c) + cfg.latent_heat_steel_j_kg + cfg.dri_reduction_endotherm_j_kg
            t_int_k = (0.65 * state.steel_temp_c + 0.35 * state.slag_temp_c) + 273.15
            t_amb_k = cfg.ambient_temp_c + 273.15
            q_wall = cfg.ua_wall_w_k * (t_int_k - t_amb_k) * dt
            q_rad = cfg.radiation_loss_factor * (1.0 - cfg.foamy_slag_loss_reduction * foam) * SIGMA * cfg.area_effective_m2 * (t_int_k**4 - t_amb_k**4) * dt
            offgas_flow = 1.25 * o2_flow + 0.78 * ng_flow + 0.4 * c_flow + 2.0
            q_offgas = offgas_flow * cfg.cp_offgas_j_kgk * max(0.0, state.offgas_temp_c - cfg.ambient_temp_c) * dt * 0.16
            q_losses = max(0.0, q_wall + max(0.0, q_rad) + q_offgas)

            q_available = max(0.0, q_arc_useful + 0.55 * q_burn + 0.6 * (q_oxy + q_c) - 0.3 * q_losses)
            melt_scrap = min(state.solid_scrap_kg, q_available / max(q_need_scrap, 1e-9))
            q_left = q_available - melt_scrap * q_need_scrap
            melt_dri = min(state.solid_dri_kg, max(0.0, q_left) / max(q_need_dri, 1e-9))
            q_melt = melt_scrap * q_need_scrap + melt_dri * q_need_dri
            state.solid_scrap_kg -= melt_scrap
            state.solid_dri_kg -= melt_dri
            state.liquid_steel_kg += melt_scrap + melt_dri * cfg.dri_fe_metallization
            state.slag_kg += flux_flow * dt * cfg.flux_to_slag_factor + oxide + melt_dri * (1.0 - cfg.dri_fe_metallization)

            q_slag_to_bath = cfg.slag_to_bath_heat_coeff_w_k * (state.slag_temp_c - state.steel_temp_c) * dt
            steel_cap = max(cfg.cp_steel_j_kgk * max(state.liquid_steel_kg, 12000.0), 1e-9)
            slag_cap = max(cfg.cp_slag_j_kgk * max(state.slag_kg, 4000.0), 1e-9)
            gas_cap = max(offgas_flow * cfg.cp_offgas_j_kgk * dt + 2.5e5, 1e-9)
            state.steel_temp_c += (q_arc_useful + 0.55 * q_burn + 0.5 * (q_oxy + q_c) + q_slag_to_bath - q_melt - 0.55 * q_losses) / steel_cap
            state.slag_temp_c += (0.25 * q_burn + 0.25 * (q_oxy + q_c) - q_slag_to_bath - 0.2 * q_losses) / slag_cap
            state.offgas_temp_c += (0.25 * q_burn + 0.25 * (q_oxy + q_c) + 0.35 * q_losses - q_offgas) / gas_cap
            state.offgas_temp_c = clamp(state.offgas_temp_c, cfg.ambient_temp_c, cfg.max_offgas_temp_c)

            tapped = start_or_continue_tapping(state, cfg)
            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += q_arc_useful + 0.55 * q_burn + 0.6 * (q_oxy + q_c)
            state.cum_losses_j += q_losses
            state.cum_oxygen_nm3 += o2_flow * dt
            state.cum_ng_nm3 += ng_flow * dt
            state.cum_carbon_kg += c_flow * dt
            return {
                "stage": stg,
                "foamy_factor": foam,
                "eta_arc": eta,
                "q_useful_mw": (q_arc_useful + 0.55 * q_burn + 0.6 * (q_oxy + q_c)) / dt / 1e6,
                "q_melt_mw": q_melt / dt / 1e6,
                "q_loss_mw": q_losses / dt / 1e6,
                "tapped_kg_s": tapped / dt,
            }

        return self.run_loop(step)
