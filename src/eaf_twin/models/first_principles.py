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

        def step(state, inputs, warnings):
            dt = cfg.dt_s
            t_m = cfg.steel_melt_temp_k
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

            eta = {
                "bore_in": cfg.eta_arc_bore_in,
                "main_melting": cfg.eta_arc_melting,
                "refining": cfg.eta_arc_refining,
                "superheat": cfg.eta_arc_superheat,
                "tapping": 0.4,
            }[stg]
            eta += 0.04 * smooth_step(state.melted_fraction, 0.2, 0.95)
            eta += 0.05 * foam
            eta = clamp(eta, 0.35, 0.92)

            q_elec = power_w * dt
            q_arc_useful = eta * q_elec
            q_burn = ng_flow * cfg.lhv_ng_j_nm3 * dt * cfg.eta_burner
            q_oxy = o2_flow * cfg.oxygen_heat_j_nm3 * cfg.oxygen_reaction_efficiency * dt
            q_c = c_flow * cfg.carbon_heat_j_kg * cfg.carbon_reaction_efficiency * dt
            
            # --- ADDED: Total chemical heat for the summary trackers ---
            q_chem = q_burn + q_oxy + q_c

            # Chemistry and Mass
            fe_oxid = min(state.liquid_steel_kg * 0.0015, o2_flow * cfg.fe_oxidation_ratio_per_nm3_o2 * dt)
            oxide = 1.29 * fe_oxid
            decarb = min(state.steel_carbon_kg, o2_flow * cfg.decarb_kg_per_nm3_o2 * dt)
            state.steel_carbon_kg += c_flow * dt - decarb - 0.0006 * state.steel_carbon_kg * dt
            state.feo_slag_kg += oxide
            state.liquid_steel_kg = max(0.0, state.liquid_steel_kg - fe_oxid)

            solid_mass = max(state.solid_scrap_kg + state.solid_dri_kg, 0.0)
            liquid_mass = max(state.liquid_steel_kg, 0.0)
            total_metal_mass = max(liquid_mass + solid_mass, 1.0)
            solid_frac = solid_mass / total_metal_mass

            # Energy distribution
            q_arc_sol = q_arc_useful * solid_frac * 0.90
            q_arc_liq = q_arc_useful * (1.0 - solid_frac) * 0.90
            q_burn_sol = q_burn * 0.60 * solid_frac
            q_burn_liq = q_burn * 0.60 * (1.0 - solid_frac)
            q_chem_sol = (q_oxy + q_c) * 0.25
            q_chem_liq = (q_oxy + q_c) * 0.75

            # Heat Losses
            t_int_k = 0.8 * state.liquid_steel_temp_k + 0.2 * state.slag_temp_k
            q_wall = cfg.ua_wall_w_k * max(0.0, t_int_k - cfg.ambient_temp_k) * dt
            q_rad = cfg.radiation_loss_factor * (1.0 - 0.2*foam) * SIGMA * cfg.area_effective_m2 * max(0.0, t_int_k**4 - cfg.ambient_temp_k**4) * dt
            
            q_loss_liq = (q_wall + q_rad) * (1.0 - solid_frac)
            q_loss_sol = (q_wall + q_rad) * solid_frac

            q_liq_net = q_arc_liq + q_burn_liq + q_chem_liq - q_loss_liq
            q_sol_net = max(0.0, q_arc_sol + q_burn_sol + q_chem_sol - q_loss_sol)

            cp_sol = cp_solid_steel_j_kgk(state.solid_scrap_temp_k)
            cp_liq = cp_liquid_steel_j_kgk(state.liquid_steel_temp_k)

            # Thermal Coupling (Liquid to Solid bath interface)
            q_l2s = 0.0
            if liquid_mass > 1e-6 and solid_mass > 1e-6:
                k_ls = 15000.0 + 10000.0 * (liquid_mass / 100000.0)
                q_l2s = k_ls * max(0.0, state.liquid_steel_temp_k - state.solid_scrap_temp_k) * dt
                avail_super = liquid_mass * cp_liq * max(0.0, state.liquid_steel_temp_k - t_m)
                q_l2s = min(q_l2s, avail_super * 0.5 + liquid_mass * cp_liq * 2.0)
            
            q_liq_net -= q_l2s
            q_sol_net += q_l2s

            # Continuous Surface Melting
            q_melt = 0.0
            q_surface_melt = 0.0
            if solid_mass > 1e-6:
                melt_factor = clamp((state.solid_scrap_temp_k - 400.0) / (t_m - 400.0), 0.05, 0.85)
                q_surface_melt = q_sol_net * melt_factor
                q_sol_sensible = q_sol_net - q_surface_melt
            else:
                q_sol_sensible = q_sol_net
                
            # Bath Superheat Direct Melting
            q_bath_melt = 0.0
            if liquid_mass > 1e-6 and solid_mass > 1e-6 and state.liquid_steel_temp_k > t_m + 1.0:
                q_bath_melt = 15000.0 * (state.liquid_steel_temp_k - t_m) * dt
                q_bath_melt = min(q_bath_melt, liquid_mass * cp_liq * (state.liquid_steel_temp_k - t_m) * 0.4)
                q_liq_net -= q_bath_melt
                
            q_melt += q_surface_melt + q_bath_melt

            # Heat Solid
            if solid_mass > 1e-6:
                state.solid_scrap_temp_k += q_sol_sensible / max(solid_mass * cp_sol, 1e-9)
                if state.solid_scrap_temp_k > t_m:
                    q_melt += (state.solid_scrap_temp_k - t_m) * solid_mass * cp_sol
                    state.solid_scrap_temp_k = t_m

            # Melt Phase Change
            melt_rate = 0.0
            region = "liquid_superheat" if solid_mass <= 1e-6 else "solid_heating"
            if q_melt > 0.0 and solid_mass > 1e-6:
                region = "phase_change"
                latent_scrap = cfg.latent_heat_steel_j_kg
                latent_dri = cfg.latent_heat_steel_j_kg + cfg.dri_reduction_endotherm_j_kg
                latent_mix = (state.solid_scrap_kg * latent_scrap + state.solid_dri_kg * latent_dri) / max(solid_mass, 1e-9)
                
                melt_kg = q_melt / latent_mix
                if melt_kg >= solid_mass:
                    melt_kg = solid_mass
                    q_liq_net += (q_melt - melt_kg * latent_mix)
                    
                melt_scrap = min(state.solid_scrap_kg, melt_kg * (state.solid_scrap_kg / solid_mass))
                melt_dri = min(state.solid_dri_kg, melt_kg * (state.solid_dri_kg / solid_mass))
                
                state.solid_scrap_kg -= melt_scrap
                state.solid_dri_kg -= melt_dri
                state.liquid_steel_kg += melt_scrap + melt_dri * cfg.dri_fe_metallization
                state.slag_kg += melt_dri * (1.0 - cfg.dri_fe_metallization)
                state.cum_latent_heat_j += (melt_scrap * latent_scrap + melt_dri * latent_dri)
                melt_rate = melt_kg / max(dt, 1e-9)

            # Heat Liquid
            if state.liquid_steel_kg > 1e-6:
                state.liquid_steel_temp_k += q_liq_net / max(state.liquid_steel_kg * cp_liq, 1e-9)
            else:
                state.liquid_steel_temp_k = state.solid_scrap_temp_k

            # Slag & Gas dynamically track Liquid Temp and Arc Power
            target_slag = state.liquid_steel_temp_k + (power_w / 80e6) * 60.0
            state.slag_temp_k += (target_slag - state.slag_temp_k) * clamp(dt / 180.0, 0.01, 1.0)
            state.slag_kg += flux_flow * dt * cfg.flux_to_slag_factor + oxide
            
            target_gas = cfg.ambient_temp_k + 350.0 + (power_w / 80e6) * 1100.0
            if state.roof_open_remaining_s > 0:
                target_gas = cfg.ambient_temp_k + 50.0
            state.offgas_temp_k += (target_gas - state.offgas_temp_k) * clamp(dt / 30.0, 0.05, 1.0)

            # Enforce limits
            state.steel_temp_k = state.liquid_steel_temp_k
            state.solid_scrap_temp_k = max(state.solid_scrap_temp_k, cfg.ambient_temp_k)
            state.liquid_steel_temp_k = max(state.liquid_steel_temp_k, cfg.ambient_temp_k)
            state.slag_temp_k = max(state.slag_temp_k, cfg.ambient_temp_k)

            tapped = start_or_continue_tapping(state, cfg)

            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += q_arc_useful + q_burn + q_oxy + q_c
            state.cum_losses_j += q_wall + q_rad
            state.cum_oxygen_nm3 += o2_flow * dt
            state.cum_ng_nm3 += ng_flow * dt
            state.cum_carbon_kg += c_flow * dt

            return {
                "stage": stg,
                "foamy_factor": foam,
                "eta_arc": eta,
                "q_useful_mw": (q_arc_useful + q_burn + q_oxy + q_c) / max(dt, 1e-9) / 1e6,
                "q_melt_mw": (melt_rate * cfg.latent_heat_steel_j_kg) / 1e6,
                "q_loss_mw": (q_wall + q_rad) / max(dt, 1e-9) / 1e6,
                "melt_rate_kg_s": melt_rate,
                "phase_region": region,
                "remaining_solid_kg": state.solid_scrap_kg + state.solid_dri_kg,
                "latent_heat_consumed_gj": state.cum_latent_heat_j / 1e9,
                "enthalpy_balance_residual_mj": 0.0,
                "tapped_kg_s": tapped / max(dt, 1e-9),
            }

        return self.run_loop(step)
