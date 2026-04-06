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
            
            # Energy inputs
            power_w = inputs["power_mw"] * 1e6
            q_elec = power_w * dt
            q_burn = cfg.eta_burner * (inputs["ng_nm3_min"] / 60.0) * cfg.lhv_ng_j_nm3 * dt
            q_oxy = 0.62 * cfg.oxygen_reaction_efficiency * (inputs["oxygen_nm3_min"] / 60.0) * cfg.oxygen_heat_j_nm3 * dt
            
            inj_c = (inputs["carbon_kg_min"] / 60.0) * dt
            q_carbon = cfg.carbon_reaction_efficiency * inj_c * cfg.carbon_heat_j_kg * dt
            
            q_chem = q_burn + q_oxy + q_carbon

            # Empirical useful heat capture rates based on stage
            eta = {"bore_in": 0.60, "main_melting": 0.75, "refining": 0.65, "superheat": 0.62, "tapping": 0.40}[stg]
            useful = eta * q_elec + 0.75 * q_burn + q_oxy + 0.75 * q_carbon

            # Simplified heat losses
            t_int_k = 0.7 * state.liquid_steel_temp_k + 0.3 * state.slag_temp_k
            amb_k = cfg.ambient_temp_k
            q_wall = cfg.ua_wall_w_k * max(0.0, t_int_k - amb_k) * dt
            q_rad = cfg.radiation_loss_factor * SIGMA * cfg.area_effective_m2 * (t_int_k**4 - amb_k**4) * dt
            q_loss = max(0.0, q_wall + max(0.0, q_rad))

            solid_mass = state.solid_scrap_kg + state.solid_dri_kg
            total_mass = max(state.liquid_steel_kg + solid_mass, 1.0)
            
            cp_solid = cfg.cp_scrap_j_kgk
            cp_liquid = cfg.cp_steel_j_kgk
            latent = cfg.latent_heat_steel_j_kg

            q_net = max(0.0, useful - q_loss)
            q_melt = 0.0
            melt_rate_kg_s = 0.0
            region = "liquid_superheat"

            # Empirical Phase Logic
            if solid_mass > 1e-6:
                if state.solid_scrap_temp_k < cfg.steel_melt_temp_k - 0.5:
                    region = "solid_heating"
                    heat_cap_solid = max(solid_mass * cp_solid, 1e-9)
                    heat_cap_liquid = max(state.liquid_steel_kg * cp_liquid, 1e-9)
                    
                    # Empirically split net heat: 80% to heat solids, 20% to heat the liquid bath
                    q_solid = q_net * 0.80
                    q_liquid = q_net * 0.20
                    
                    state.solid_scrap_temp_k += q_solid / heat_cap_solid
                    state.liquid_steel_temp_k += q_liquid / heat_cap_liquid
                    
                    # Prevent overshooting melt point; carry excess to melting
                    if state.solid_scrap_temp_k > cfg.steel_melt_temp_k:
                        excess = (state.solid_scrap_temp_k - cfg.steel_melt_temp_k) * heat_cap_solid
                        state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                        melt_scrap = min(state.solid_scrap_kg, excess / max(latent, 1e-9))
                        q_melt = melt_scrap * latent
                        state.solid_scrap_kg -= melt_scrap
                        state.liquid_steel_kg += melt_scrap
                else:
                    region = "phase_change"
                    state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                    
                    # If the liquid bath naturally heated beyond the melt point, drain that heat back to melting scrap
                    bath_excess_heat = 0.0
                    if state.liquid_steel_temp_k > cfg.steel_melt_temp_k + 2.0:
                        bath_excess_heat = (state.liquid_steel_temp_k - cfg.steel_melt_temp_k) * state.liquid_steel_kg * cp_liquid * 0.25
                        state.liquid_steel_temp_k -= (bath_excess_heat / max(state.liquid_steel_kg * cp_liquid, 1e-9))
                        
                    q_for_melt = q_net + bath_excess_heat
                    
                    melt_scrap = min(state.solid_scrap_kg, q_for_melt / max(latent, 1e-9))
                    q_for_melt -= melt_scrap * latent
                    
                    q_need_dri = latent + cfg.dri_reduction_endotherm_j_kg
                    melt_dri = min(state.solid_dri_kg, max(0.0, q_for_melt) / max(q_need_dri, 1e-9))
                    
                    q_melt = melt_scrap * latent + melt_dri * q_need_dri
                    melt_rate_kg_s = (melt_scrap + melt_dri) / max(dt, 1e-9)
                    
                    state.solid_scrap_kg -= melt_scrap
                    state.solid_dri_kg -= melt_dri
                    state.liquid_steel_kg += melt_scrap + melt_dri * cfg.dri_fe_metallization
                    state.slag_kg += melt_dri * (1.0 - cfg.dri_fe_metallization)
            else:
                region = "liquid_superheat"
                state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                state.solid_scrap_kg = 0.0
                state.solid_dri_kg = 0.0
                state.liquid_steel_temp_k += q_net / max(total_mass * cp_liquid, 1e-9)

            # Empirical Slag Temp (tracks slightly hotter than bath dependent on arc/oxy intensity)
            power_ratio = inputs["power_mw"] / 80.0
            oxy_ratio = inputs["oxygen_nm3_min"] / 80.0
            target_slag = state.liquid_steel_temp_k + 20.0 + 40.0 * power_ratio + 30.0 * oxy_ratio
            state.slag_temp_k += 0.08 * (target_slag - state.slag_temp_k)
            
            # Empirical Offgas Temp (responds fast to chemical heat and power)
            target_gas = cfg.ambient_temp_k + 250.0 + 1000.0 * (q_chem / max(dt, 1e-9) / 1e6 / 20.0) + 400.0 * power_ratio
            state.offgas_temp_k += 0.12 * (target_gas - state.offgas_temp_k)
            state.offgas_temp_k = clamp(state.offgas_temp_k, cfg.ambient_temp_k, cfg.max_offgas_temp_k)

            # Flow rates & Basic masses
            state.slag_kg += inputs["flux_kg_min"] / 60.0 * dt * cfg.flux_to_slag_factor
            decarb = min(state.steel_carbon_kg, inputs["oxygen_nm3_min"] / 60.0 * dt * cfg.decarb_kg_per_nm3_o2 * 0.7)
            state.steel_carbon_kg += inj_c - decarb
            
            state.steel_temp_k = state.liquid_steel_temp_k
            state.solid_scrap_temp_k = max(state.solid_scrap_temp_k, cfg.ambient_temp_k)
            state.liquid_steel_temp_k = max(state.liquid_steel_temp_k, cfg.ambient_temp_k)
            state.slag_temp_k = max(state.slag_temp_k, cfg.ambient_temp_k)
            
            tapped = start_or_continue_tapping(state, cfg)

            # Log accumulators
            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += useful
            state.cum_losses_j += q_loss
            state.cum_oxygen_nm3 += inputs["oxygen_nm3_min"] / 60.0 * dt
            state.cum_ng_nm3 += inputs["ng_nm3_min"] / 60.0 * dt
            state.cum_carbon_kg += inj_c
            
            return {
                "stage": stg,
                "q_useful_mw": useful / max(dt, 1e-9) / 1e6,
                "q_melt_mw": q_melt / max(dt, 1e-9) / 1e6,
                "q_loss_mw": q_loss / max(dt, 1e-9) / 1e6,
                "melt_rate_kg_s": melt_rate_kg_s,
                "phase_region": region,
                "tapped_kg_s": tapped / max(dt, 1e-9),
            }

        return self.run_loop(step)
