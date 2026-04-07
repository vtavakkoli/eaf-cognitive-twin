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
            q_chem = q_burn + q_oxy + q_c

            # Chemistry and mass effects (coupled to oxygen and carbon inputs).
            fe_oxid = min(state.liquid_steel_kg * 0.0015, o2_flow * cfg.fe_oxidation_ratio_per_nm3_o2 * dt)
            oxide = 1.29 * fe_oxid
            decarb = min(state.steel_carbon_kg, o2_flow * cfg.decarb_kg_per_nm3_o2 * dt)
            state.steel_carbon_kg += c_flow * dt - decarb - 0.0006 * state.steel_carbon_kg * dt
            state.feo_slag_kg += oxide
            state.liquid_steel_kg = max(0.0, state.liquid_steel_kg - fe_oxid)

            # Heat losses (all in Kelvin).
            t_int_k = 0.7 * state.liquid_steel_temp_k + 0.3 * state.slag_temp_k
            q_wall = cfg.ua_wall_w_k * max(0.0, t_int_k - cfg.ambient_temp_k) * dt
            q_rad = (
                cfg.radiation_loss_factor
                * (1.0 - cfg.foamy_slag_loss_reduction * foam)
                * SIGMA
                * cfg.area_effective_m2
                * max(0.0, t_int_k**4 - cfg.ambient_temp_k**4)
                * dt
            )
            offgas_flow = 1.25 * o2_flow + 0.78 * ng_flow + 0.4 * c_flow + 2.0
            effective_gas_mass_flow = offgas_flow + 24.0
            gas_source_w = (0.17 * power_w + 0.20 * (q_chem / max(dt, 1e-9)))
            target_gas_k = cfg.ambient_temp_k + gas_source_w / max(effective_gas_mass_flow * cfg.cp_offgas_j_kgk, 1e-6)
            if state.roof_open_remaining_s > 1e-6:
                target_gas_k = cfg.ambient_temp_k
            state.offgas_temp_k += (target_gas_k - state.offgas_temp_k) * clamp(dt / 30.0, 0.05, 1.0)
            state.offgas_temp_k = clamp(state.offgas_temp_k, cfg.ambient_temp_k, cfg.max_offgas_temp_k)
            q_offgas = offgas_flow * cfg.cp_offgas_j_kgk * max(0.0, state.offgas_temp_k - cfg.ambient_temp_k) * dt
            q_losses = q_wall + q_rad + q_offgas

            solid_mass = max(state.solid_scrap_kg + state.solid_dri_kg, 0.0)
            liquid_mass = max(state.liquid_steel_kg, 0.0)
            total_metal_mass = max(liquid_mass + solid_mass, 1.0)
            solid_fraction = solid_mass / total_metal_mass

            # Energy routing 
            q_arc_to_metal = q_arc_useful * (0.88 if not self.enhanced else 0.90)
            q_arc_to_slag = q_arc_useful * (0.10 if not self.enhanced else 0.08)
            q_arc_to_solid = q_arc_to_metal * solid_fraction
            q_arc_to_liquid = q_arc_to_metal * (1.0 - solid_fraction)

            q_burn_to_solid = q_burn * 0.38
            q_burn_to_liquid = q_burn * 0.30
            q_burn_to_slag = q_burn * 0.20

            q_chem_to_solid = 0.20 * (q_oxy + q_c)
            q_chem_to_liquid = 0.45 * (q_oxy + q_c)
            q_chem_to_slag = 0.30 * (q_oxy + q_c)

            q_slag_to_bath = cfg.slag_to_bath_heat_coeff_w_k * (state.slag_temp_k - state.liquid_steel_temp_k) * dt
            q_flux_sink = flux_flow * dt * cfg.cp_slag_j_kgk * max(0.0, state.slag_temp_k - cfg.ambient_temp_k)

            cp_sol = cp_solid_steel_j_kgk(state.solid_scrap_temp_k)
            cp_liq = cp_liquid_steel_j_kgk(state.liquid_steel_temp_k)

            # --- CONTINUOUS MELTING & THERMAL COUPLING ---
            # 1. External heat direct to solid goes entirely into surface melting
            q_direct_melt = q_arc_to_solid + q_burn_to_solid + q_chem_to_solid

            # 2. Strong thermal interphase coupling between liquid and solid bulk
            q_interphase = 0.0
            q_bath_melt = 0.0
            if solid_mass > 1e-6 and liquid_mass > 1e-6:
                h_interphase = 45000.0 + 15000.0 * (liquid_mass / 100000.0)
                q_interphase = h_interphase * (state.liquid_steel_temp_k - state.solid_scrap_temp_k) * dt
                
                # Bath also melts solid directly if it contains superheat
                if state.liquid_steel_temp_k > t_m:
                    q_bath_melt = 25000.0 * (state.liquid_steel_temp_k - t_m) * dt
                    q_bath_melt = min(q_bath_melt, liquid_mass * cp_liq * (state.liquid_steel_temp_k - t_m))

            q_liquid_net = q_arc_to_liquid + q_burn_to_liquid + q_chem_to_liquid + q_slag_to_bath - 0.35 * q_losses - q_interphase - q_bath_melt
            q_slag_net = q_arc_to_slag + q_burn_to_slag + q_chem_to_slag - q_slag_to_bath - q_flux_sink - 0.25 * q_losses

            q_melt_total = q_direct_melt + q_bath_melt
            q_solid_sensible = q_interphase

            region = "liquid_superheat"
            melt_scrap = 0.0
            melt_dri = 0.0
            q_actual_melt = 0.0

            if solid_mass > 1e-6:
                # Process Sensible Bulk Heating
                sensible_need = solid_mass * cp_sol * max(0.0, t_m - state.solid_scrap_temp_k)
                if q_solid_sensible > sensible_need:
                    q_melt_total += (q_solid_sensible - sensible_need)
                    q_solid_sensible = sensible_need
                    state.solid_scrap_temp_k = t_m
                    region = "phase_change"
                else:
                    state.solid_scrap_temp_k += q_solid_sensible / max(solid_mass * cp_sol, 1e-9)
                    region = "solid_heating"

                # Process Melting
                if q_melt_total > 0.0:
                    latent_scrap = cfg.latent_heat_steel_j_kg
                    latent_dri = cfg.latent_heat_steel_j_kg + cfg.dri_reduction_endotherm_j_kg
                    latent_mix = (state.solid_scrap_kg * latent_scrap + state.solid_dri_kg * latent_dri) / max(solid_mass, 1e-9)
                    
                    melt_total_kg = min(solid_mass, q_melt_total / max(latent_mix, 1e-9))
                    
                    if melt_total_kg > 0.0:
                        scrap_share = state.solid_scrap_kg / max(solid_mass, 1e-9)
                        dri_share = state.solid_dri_kg / max(solid_mass, 1e-9)
                        melt_scrap = melt_total_kg * scrap_share
                        melt_dri = melt_total_kg * dri_share
                        
                        state.solid_scrap_kg -= melt_scrap
                        state.solid_dri_kg -= melt_dri
                        state.liquid_steel_kg += melt_scrap + melt_dri * cfg.dri_fe_metallization
                        state.slag_kg += melt_dri * (1.0 - cfg.dri_fe_metallization)
                        
                        q_actual_melt = melt_scrap * latent_scrap + melt_dri * latent_dri
                        state.cum_latent_heat_j += q_actual_melt
                        
                    # Return unused melt energy back to the liquid bath
                    q_unused = max(0.0, q_melt_total - q_actual_melt)
                    q_liquid_net += q_unused

            if state.solid_scrap_kg + state.solid_dri_kg <= 1e-6:
                state.solid_scrap_kg = 0.0
                state.solid_dri_kg = 0.0
                state.solid_scrap_temp_k = t_m
                region = "liquid_superheat"

            # Liquid update
            liquid_mass = max(state.liquid_steel_kg, 0.0)
            if liquid_mass > 1e-6:
                state.liquid_steel_temp_k += q_liquid_net / max(liquid_mass * cp_liq, 1e-9)
            else:
                state.liquid_steel_temp_k = state.solid_scrap_temp_k

            # Slag update and mass additions.
            state.slag_kg += flux_flow * dt * cfg.flux_to_slag_factor + oxide
            eff_cap_slag = max(state.slag_kg * cfg.cp_slag_j_kgk, 2500.0 * cfg.cp_slag_j_kgk)
            state.slag_temp_k += q_slag_net / eff_cap_slag

            state.solid_scrap_temp_k = max(state.solid_scrap_temp_k, cfg.ambient_temp_k)
            state.liquid_steel_temp_k = max(state.liquid_steel_temp_k, cfg.ambient_temp_k)
            state.slag_temp_k = max(state.slag_temp_k, cfg.ambient_temp_k)
            state.steel_temp_k = state.liquid_steel_temp_k

            tapped = start_or_continue_tapping(state, cfg)

            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += q_arc_to_metal + q_burn + q_chem_to_liquid + max(0.0, q_slag_to_bath)
            state.cum_losses_j += q_losses
            state.cum_oxygen_nm3 += o2_flow * dt
            state.cum_ng_nm3 += ng_flow * dt
            state.cum_carbon_kg += c_flow * dt

            delta_h = q_liquid_net + q_solid_sensible + q_slag_net + q_actual_melt
            state.enthalpy_balance_residual_j = (q_arc_useful + q_burn + q_oxy + q_c) - (q_losses + delta_h)
            
            if abs(state.enthalpy_balance_residual_j) > 0.08 * max(q_elec + q_chem, 1.0):
                warnings.append(f"Energy balance residual high: {state.enthalpy_balance_residual_j/1e6:.2f} MJ at t={state.time_s/60:.2f} min")
            if state.liquid_steel_kg > 1e-6 and state.liquid_steel_temp_k < t_m - 1200.0:
                warnings.append(f"Liquid steel too cold: {state.liquid_steel_temp_k:.1f} K")
            if state.melted_fraction < -1e-6 or state.melted_fraction > 1.0001:
                warnings.append(f"Melted fraction out of bounds: {state.melted_fraction:.4f}")

            return {
                "stage": stg,
                "foamy_factor": foam,
                "eta_arc": eta,
                "q_useful_mw": (q_arc_to_metal + q_burn + q_chem_to_liquid + max(0.0, q_slag_to_bath)) / max(dt, 1e-9) / 1e6,
                "q_melt_mw": q_actual_melt / max(dt, 1e-9) / 1e6,
                "q_loss_mw": q_losses / max(dt, 1e-9) / 1e6,
                "melt_rate_kg_s": (melt_scrap + melt_dri) / max(dt, 1e-9),
                "phase_region": region,
                "remaining_solid_kg": state.solid_scrap_kg + state.solid_dri_kg,
                "latent_heat_consumed_gj": state.cum_latent_heat_j / 1e9,
                "enthalpy_balance_residual_mj": state.enthalpy_balance_residual_j / 1e6,
                "tapped_kg_s": tapped / max(dt, 1e-9),
            }

        return self.run_loop(step)
