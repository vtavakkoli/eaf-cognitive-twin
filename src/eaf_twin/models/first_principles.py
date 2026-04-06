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

        def sensible_heat_liquid_steel_j_kg(t0_k: float, t1_k: float) -> float:
            t_avg = 0.5 * (t0_k + t1_k)
            return cp_liquid_steel_j_kgk(t_avg) * (t1_k - t0_k)

        def slag_sensible_enthalpy_j_kg(t0_k: float, t1_k: float) -> float:
            return cfg.cp_slag_j_kgk * (t1_k - t0_k)

        def offgas_sensible_enthalpy_j_kg(t0_k: float, t1_k: float) -> float:
            return cfg.cp_offgas_j_kgk * (t1_k - t0_k)

        def step(state, inputs, warnings):
            dt = cfg.dt_s
            stg = stage_name(state.time_s / SECONDS_PER_MIN, state.melted_fraction)
            state.steel_temp_k = state.liquid_steel_temp_k
            
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

            # Chemistry and Mass
            fe_oxid = min(state.liquid_steel_kg * 0.0015, o2_flow * cfg.fe_oxidation_ratio_per_nm3_o2 * dt)
            oxide = 1.29 * fe_oxid
            decarb = min(state.steel_carbon_kg, o2_flow * cfg.decarb_kg_per_nm3_o2 * dt)
            state.steel_carbon_kg += c_flow * dt - decarb - 0.0006 * state.steel_carbon_kg * dt
            state.feo_slag_kg += oxide
            state.liquid_steel_kg -= fe_oxid

            # Losses
            t_int_k = 0.8 * state.liquid_steel_temp_k + 0.2 * state.slag_temp_k
            q_wall = cfg.ua_wall_w_k * max(0.0, t_int_k - cfg.ambient_temp_k) * dt
            q_rad = cfg.radiation_loss_factor * (1.0 - cfg.foamy_slag_loss_reduction * foam) * SIGMA * cfg.area_effective_m2 * (t_int_k**4 - cfg.ambient_temp_k**4) * dt
            
            # Gas Temperature (First order lag correctly limits runaway to 2200C)
            target_gas_k = cfg.ambient_temp_k + 300.0 + 1200.0 * (power_w / 80e6) + 800.0 * (q_chem / max(dt, 1e-9) / 20e6)
            state.offgas_temp_k += (target_gas_k - state.offgas_temp_k) * (dt / 15.0)
            state.offgas_temp_k = clamp(state.offgas_temp_k, cfg.ambient_temp_k, cfg.max_offgas_temp_k)
            
            offgas_flow = 1.25 * o2_flow + 0.78 * ng_flow + 0.4 * c_flow + 2.0
            q_offgas = offgas_flow * cfg.cp_offgas_j_kgk * (state.offgas_temp_k - cfg.ambient_temp_k) * dt
            q_losses = q_wall + q_rad + q_offgas

            # Distribute Heat based on Melt Fraction (Shielding effect)
            f_melt = state.melted_fraction
            
            q_arc_to_slag = q_arc_useful * (0.12 if not self.enhanced else 0.10)
            q_arc_to_metal = q_arc_useful - q_arc_to_slag
            
            q_arc_to_liquid = q_arc_to_metal * (f_melt ** 1.5)
            q_arc_to_solid = q_arc_to_metal - q_arc_to_liquid
            
            q_burn_to_slag = q_burn * 0.20
            q_burn_to_metal = q_burn - q_burn_to_slag
            q_burn_to_liquid = q_burn_to_metal * f_melt
            q_burn_to_solid = q_burn_to_metal - q_burn_to_liquid
            
            q_chem_to_slag = (q_oxy + q_c) * 0.25
            q_chem_to_liquid = (q_oxy + q_c) - q_chem_to_slag

            # Slag dynamics
            q_slag_to_bath = 25000.0 * (state.slag_temp_k - state.liquid_steel_temp_k) * dt
            q_slag_net = q_arc_to_slag + q_burn_to_slag + q_chem_to_slag - q_slag_to_bath - 0.15 * q_losses
            state.slag_temp_k += q_slag_net / max(state.slag_kg * cfg.cp_slag_j_kgk, 1e-9)

            # Solid & Liquid dynamics
            q_liquid_net = q_arc_to_liquid + q_burn_to_liquid + q_chem_to_liquid + q_slag_to_bath - 0.45 * q_losses
            q_solid_net = q_arc_to_solid + q_burn_to_solid - 0.15 * q_losses

            solid_mass = state.solid_scrap_kg + state.solid_dri_kg
            cp_sol = cp_solid_steel_j_kgk(state.solid_scrap_temp_k)
            cp_liq = cp_liquid_steel_j_kgk(state.liquid_steel_temp_k)
            
            melt_scrap = 0.0
            melt_dri = 0.0
            q_melt = 0.0
            region = "liquid_superheat"

            if solid_mass > 1e-6:
                # Liquid transfers massive heat to solid if superheated
                q_convection = 0.0
                if state.liquid_steel_temp_k > state.solid_scrap_temp_k:
                    q_convection = 15000.0 * (state.liquid_steel_temp_k - state.solid_scrap_temp_k) * dt
                    q_convection = min(q_convection, 0.5 * state.liquid_steel_kg * cp_liq * (state.liquid_steel_temp_k - state.solid_scrap_temp_k))
                
                q_liquid_net -= q_convection
                q_solid_net += q_convection
                
                state.solid_scrap_temp_k += q_solid_net / max(solid_mass * cp_sol, 1e-9)
                
                if state.solid_scrap_temp_k >= cfg.steel_melt_temp_k:
                    region = "phase_change"
                    excess_j = (state.solid_scrap_temp_k - cfg.steel_melt_temp_k) * solid_mass * cp_sol
                    state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                    
                    # Force liquid's excess heat into melting the solid
                    if state.liquid_steel_temp_k > cfg.steel_melt_temp_k + 2.0:
                        bath_excess_j = (state.liquid_steel_temp_k - cfg.steel_melt_temp_k) * state.liquid_steel_kg * cp_liq * 0.8
                        excess_j += bath_excess_j
                        state.liquid_steel_temp_k -= bath_excess_j / max(state.liquid_steel_kg * cp_liq, 1e-9)
                    
                    latent = cfg.latent_heat_steel_j_kg
                    melt_scrap = min(state.solid_scrap_kg, excess_j / latent)
                    excess_j -= melt_scrap * latent
                    
                    latent_dri = latent + cfg.dri_reduction_endotherm_j_kg
                    melt_dri = min(state.solid_dri_kg, excess_j / latent_dri)
                    
                    q_melt = melt_scrap * latent + melt_dri * latent_dri
                    
                    new_liquid_kg = melt_scrap + melt_dri * cfg.dri_fe_metallization
                    if new_liquid_kg > 0:
                        # Enthalpy transfer: Liquid perfectly mixes with incoming 1535C steel
                        total_liq = state.liquid_steel_kg + new_liquid_kg
                        state.liquid_steel_temp_k = (state.liquid_steel_kg * state.liquid_steel_temp_k + new_liquid_kg * cfg.steel_melt_temp_k) / total_liq
                        
                        state.solid_scrap_kg -= melt_scrap
                        state.solid_dri_kg -= melt_dri
                        state.liquid_steel_kg = total_liq
                        state.slag_kg += melt_dri * (1.0 - cfg.dri_fe_metallization)
                else:
                    region = "solid_heating"
                    
                state.liquid_steel_temp_k += q_liquid_net / max(state.liquid_steel_kg * cp_liq, 1e-9)
            else:
                region = "liquid_superheat"
                state.solid_scrap_kg = 0.0
                state.solid_dri_kg = 0.0
                state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                # A minimum effective thermal mass prevents temperature crash when tapping drains the furnace
                eff_cap_liq = max(state.liquid_steel_kg * cp_liq, 5000.0 * cp_liq)
                state.liquid_steel_temp_k += q_liquid_net / eff_cap_liq

            state.slag_kg += flux_flow * dt * cfg.flux_to_slag_factor + oxide
            
            state.solid_scrap_temp_k = max(state.solid_scrap_temp_k, cfg.ambient_temp_k)
            state.liquid_steel_temp_k = max(state.liquid_steel_temp_k, cfg.ambient_temp_k)
            state.steel_temp_k = state.liquid_steel_temp_k
            state.slag_temp_k = max(state.slag_temp_k, cfg.ambient_temp_k)

            tapped = start_or_continue_tapping(state, cfg)
            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += q_arc_to_metal + q_burn_to_metal + q_chem_to_liquid + max(0.0, q_slag_to_bath)
            state.cum_losses_j += q_losses
            state.cum_oxygen_nm3 += o2_flow * dt
            state.cum_ng_nm3 += ng_flow * dt
            state.cum_carbon_kg += c_flow * dt
            
            return {
                "stage": stg,
                "foamy_factor": foam,
                "eta_arc": eta,
                "q_useful_mw": (q_arc_to_metal + q_burn_to_metal + q_chem_to_liquid + max(0.0, q_slag_to_bath)) / max(dt, 1e-9) / 1e6,
                "q_melt_mw": q_melt / max(dt, 1e-9) / 1e6,
                "q_loss_mw": q_losses / max(dt, 1e-9) / 1e6,
                "melt_rate_kg_s": (melt_scrap + melt_dri) / max(dt, 1e-9),
                "phase_region": region,
                "h_steel_sensible_mj": sensible_heat_liquid_steel_j_kg(cfg.ambient_temp_k, state.liquid_steel_temp_k) * state.liquid_steel_kg / 1e6,
                "h_slag_sensible_mj": slag_sensible_enthalpy_j_kg(cfg.ambient_temp_k, state.slag_temp_k) * state.slag_kg / 1e6,
                "h_offgas_sensible_mj": offgas_sensible_enthalpy_j_kg(cfg.ambient_temp_k, state.offgas_temp_k) * offgas_flow * dt / 1e6,
                "tapped_kg_s": tapped / max(dt, 1e-9),
            }

        return self.run_loop(step)
