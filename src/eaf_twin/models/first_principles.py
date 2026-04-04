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

        def cp_solid_steel_j_kgk(t_k: float) -> float:
            return cfg.cp_scrap_j_kgk * (1.0 + 8.5e-5 * (t_k - cfg.ambient_temp_k))

        def cp_liquid_steel_j_kgk(t_k: float) -> float:
            return cfg.cp_steel_j_kgk * (1.0 + 6.5e-5 * max(0.0, t_k - cfg.steel_melt_temp_k))

        def sensible_heat_solid_steel_j_kg(t0_k: float, t1_k: float) -> float:
            t_avg = 0.5 * (t0_k + t1_k)
            return cp_solid_steel_j_kgk(t_avg) * (t1_k - t0_k)

        def sensible_heat_liquid_steel_j_kg(t0_k: float, t1_k: float) -> float:
            t_avg = 0.5 * (t0_k + t1_k)
            return cp_liquid_steel_j_kgk(t_avg) * (t1_k - t0_k)

        def latent_heat_steel_j_kg() -> float:
            return cfg.latent_heat_steel_j_kg

        def slag_sensible_enthalpy_j_kg(t0_k: float, t1_k: float) -> float:
            return cfg.cp_slag_j_kgk * (t1_k - t0_k)

        def offgas_sensible_enthalpy_j_kg(t0_k: float, t1_k: float) -> float:
            return cfg.cp_offgas_j_kgk * (t1_k - t0_k)

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
            q_burn = ng_flow * cfg.lhv_ng_j_nm3 * dt
            q_oxy = o2_flow * cfg.oxygen_heat_j_nm3 * cfg.oxygen_reaction_efficiency * dt
            q_c = c_flow * cfg.carbon_heat_j_kg * cfg.carbon_reaction_efficiency * dt
            q_chem = q_burn + q_oxy + q_c

            fe_oxid = min(state.liquid_steel_kg * 0.0015, o2_flow * cfg.fe_oxidation_ratio_per_nm3_o2 * dt)
            oxide = 1.29 * fe_oxid
            decarb = min(state.steel_carbon_kg, o2_flow * cfg.decarb_kg_per_nm3_o2 * dt)
            state.steel_carbon_kg += c_flow * dt - decarb - 0.0006 * state.steel_carbon_kg * dt
            state.feo_slag_kg += oxide
            state.liquid_steel_kg -= fe_oxid

            t_int_k = 0.65 * state.steel_temp_k + 0.35 * state.slag_temp_k
            t_amb_k = cfg.ambient_temp_k
            q_wall = cfg.ua_wall_w_k * max(0.0, t_int_k - t_amb_k) * dt
            q_rad = cfg.radiation_loss_factor * (1.0 - cfg.foamy_slag_loss_reduction * foam) * SIGMA * cfg.area_effective_m2 * (t_int_k**4 - t_amb_k**4) * dt
            offgas_flow = 1.25 * o2_flow + 0.78 * ng_flow + 0.4 * c_flow + 2.0
            q_offgas = offgas_flow * cfg.cp_offgas_j_kgk * max(0.0, state.offgas_temp_k - t_amb_k) * dt * 0.16
            q_losses = max(0.0, q_wall + max(0.0, q_rad) + q_offgas)

            q_arc_to_metal = q_arc_useful * (0.88 if not self.enhanced else 0.90)
            q_arc_to_slag = q_arc_useful * 0.10
            q_arc_to_gas = q_arc_useful - q_arc_to_metal - q_arc_to_slag
            q_burn_to_metal = q_burn * (0.50 if not self.enhanced else 0.54)
            q_burn_to_slag = q_burn * 0.18
            q_burn_to_gas = q_burn - q_burn_to_metal - q_burn_to_slag
            q_chem_to_metal = 0.70 * (q_oxy + q_c)
            q_chem_to_slag = 0.16 * (q_oxy + q_c)
            q_chem_to_gas = 0.14 * (q_oxy + q_c)

            q_slag_to_bath = cfg.slag_to_bath_heat_coeff_w_k * (state.slag_temp_k - state.steel_temp_k) * dt
            q_metal_net = q_arc_to_metal + q_burn_to_metal + q_chem_to_metal + q_slag_to_bath - (0.2 if not self.enhanced else 0.16) * q_losses
            q_slag_net = q_arc_to_slag + q_burn_to_slag + q_chem_to_slag - q_slag_to_bath - 0.25 * q_losses
            q_gas_net = q_arc_to_gas + q_burn_to_gas + q_chem_to_gas + 0.15 * q_losses - q_offgas

            melt_scrap = 0.0
            melt_dri = 0.0
            q_melt = 0.0
            melt_rate_kg_s = 0.0

            solid_mass = state.solid_scrap_kg + state.solid_dri_kg
            total_metal_mass = max(state.liquid_steel_kg + solid_mass, 1.0)
            cp_solid = cp_solid_steel_j_kgk(state.steel_temp_k)
            cp_liquid = cp_liquid_steel_j_kgk(state.steel_temp_k)
            h_melt_start = sensible_heat_solid_steel_j_kg(cfg.ambient_temp_k, cfg.steel_melt_temp_k)
            h_melt_end = h_melt_start + latent_heat_steel_j_kg()
            if state.steel_temp_k < cfg.steel_melt_temp_k:
                h_metal = cp_solid * (state.steel_temp_k - cfg.ambient_temp_k)
            else:
                h_metal = h_melt_end + cp_liquid * (state.steel_temp_k - cfg.steel_melt_temp_k)

            region = "liquid_superheat"
            if solid_mass > 1e-6 and h_metal < h_melt_start:
                region = "solid_heating"
                if q_metal_net != 0.0:
                    state.steel_temp_k += q_metal_net / max(total_metal_mass * cp_solid, 1e-9)
            elif solid_mass > 1e-6 and h_metal <= h_melt_end:
                region = "phase_change"
                q_need_scrap = latent_heat_steel_j_kg()
                q_need_dri = latent_heat_steel_j_kg() + cfg.dri_reduction_endotherm_j_kg
                q_for_melt = max(0.0, q_metal_net)
                melt_scrap = min(state.solid_scrap_kg, q_for_melt / max(q_need_scrap, 1e-9))
                q_for_melt -= melt_scrap * q_need_scrap
                melt_dri = min(state.solid_dri_kg, q_for_melt / max(q_need_dri, 1e-9))
                q_for_melt -= melt_dri * q_need_dri
                q_melt = melt_scrap * q_need_scrap + melt_dri * q_need_dri
                melt_rate_kg_s = (melt_scrap + melt_dri) / max(dt, 1e-9)
                state.solid_scrap_kg -= melt_scrap
                state.solid_dri_kg -= melt_dri
                state.liquid_steel_kg += melt_scrap + melt_dri * cfg.dri_fe_metallization
                state.slag_kg += melt_dri * (1.0 - cfg.dri_fe_metallization)
                if state.solid_scrap_kg + state.solid_dri_kg > 1e-6:
                    state.steel_temp_k = clamp(state.steel_temp_k, cfg.steel_melt_temp_k - 8.0, cfg.steel_melt_temp_k + 12.0)
            else:
                region = "liquid_superheat"
                if solid_mass > 1e-6:
                    state.solid_scrap_kg = 0.0
                    state.solid_dri_kg = 0.0
                if q_metal_net != 0.0:
                    state.steel_temp_k += q_metal_net / max(total_metal_mass * cp_liquid, 1e-9)

            state.slag_kg += flux_flow * dt * cfg.flux_to_slag_factor + oxide

            slag_cap = max(state.slag_kg * cfg.cp_slag_j_kgk, 1e-9)
            state.slag_temp_k += q_slag_net / slag_cap

            gas_cap = max(offgas_flow * cfg.cp_offgas_j_kgk * dt + 2.5e5, 1e-9)
            state.offgas_temp_k += q_gas_net / gas_cap
            state.offgas_temp_k = clamp(state.offgas_temp_k, cfg.ambient_temp_k, cfg.max_offgas_temp_k)

            if state.solid_scrap_kg + state.solid_dri_kg > 1e-6:
                state.steel_temp_k = min(state.steel_temp_k, cfg.steel_melt_temp_k + 20.0)
            state.steel_temp_k = max(state.steel_temp_k, cfg.ambient_temp_k)
            state.slag_temp_k = max(state.slag_temp_k, cfg.ambient_temp_k)

            if state.melted_fraction >= 0.999 and state.steel_temp_k < cfg.steel_melt_temp_k - 1.0:
                warnings.append("Inconsistent state: full melt reached below steel melting range; clamped.")
                state.steel_temp_k = cfg.steel_melt_temp_k

            tapped = start_or_continue_tapping(state, cfg)
            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += q_arc_to_metal + q_burn_to_metal + q_chem_to_metal + max(0.0, q_slag_to_bath)
            state.cum_losses_j += q_losses
            state.cum_oxygen_nm3 += o2_flow * dt
            state.cum_ng_nm3 += ng_flow * dt
            state.cum_carbon_kg += c_flow * dt
            return {
                "stage": stg,
                "foamy_factor": foam,
                "eta_arc": eta,
                "q_useful_mw": (q_arc_to_metal + q_burn_to_metal + q_chem_to_metal + max(0.0, q_slag_to_bath)) / dt / 1e6,
                "q_melt_mw": q_melt / dt / 1e6,
                "q_loss_mw": q_losses / dt / 1e6,
                "melt_rate_kg_s": melt_rate_kg_s,
                "phase_region": region,
                "h_steel_sensible_mj": sensible_heat_liquid_steel_j_kg(cfg.ambient_temp_k, state.steel_temp_k) * state.liquid_steel_kg / 1e6,
                "h_slag_sensible_mj": slag_sensible_enthalpy_j_kg(cfg.ambient_temp_k, state.slag_temp_k) * state.slag_kg / 1e6,
                "h_offgas_sensible_mj": offgas_sensible_enthalpy_j_kg(cfg.ambient_temp_k, state.offgas_temp_k) * offgas_flow * dt / 1e6,
                "tapped_kg_s": tapped / dt,
            }

        return self.run_loop(step)
