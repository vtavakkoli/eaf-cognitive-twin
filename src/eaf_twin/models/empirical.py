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
            q_burn = cfg.eta_burner * (inputs["ng_nm3_min"] / 60.0) * cfg.lhv_ng_j_nm3 * dt
            q_oxy = 0.62 * cfg.oxygen_reaction_efficiency * (inputs["oxygen_nm3_min"] / 60.0) * cfg.oxygen_heat_j_nm3 * dt
            
            inj_c = (inputs["carbon_kg_min"] / 60.0) * dt
            q_carbon = cfg.carbon_reaction_efficiency * inj_c * cfg.carbon_heat_j_kg * dt
            q_chem = q_burn + q_oxy + q_carbon

            eta = {"bore_in": 0.60, "main_melting": 0.75, "refining": 0.65, "superheat": 0.62, "tapping": 0.40}[stg]
            useful = eta * q_elec + 0.75 * q_burn + q_oxy + 0.75 * q_carbon

            t_int_k = 0.7 * state.liquid_steel_temp_k + 0.3 * state.slag_temp_k
            amb_k = cfg.ambient_temp_k
            q_wall = cfg.ua_wall_w_k * max(0.0, t_int_k - amb_k) * dt
            q_rad = cfg.radiation_loss_factor * SIGMA * cfg.area_effective_m2 * (t_int_k**4 - amb_k**4) * dt
            q_loss = max(0.0, q_wall + max(0.0, q_rad))

            solid_mass = state.solid_scrap_kg + state.solid_dri_kg
            cp_solid = cfg.cp_scrap_j_kgk
            cp_liquid = cfg.cp_steel_j_kgk
            latent = cfg.latent_heat_steel_j_kg
            
            cap_sol = solid_mass * cp_solid
            cap_liq = state.liquid_steel_kg * cp_liquid

            q_net = max(0.0, useful - q_loss)
            q_melt = 0.0
            melt_rate_kg_s = 0.0
            region = "liquid_superheat"

            if solid_mass > 1e-6:
                # 1. Thermal equilibration 
                if state.liquid_steel_kg > 1e-6:
                    t_eq = (cap_liq * state.liquid_steel_temp_k + cap_sol * state.solid_scrap_temp_k) / max(cap_liq + cap_sol, 1e-9)
                    mix_fraction = min(1.0, dt / 60.0)
                    q_transfer = mix_fraction * cap_liq * (state.liquid_steel_temp_k - t_eq) / dt
                else:
                    q_transfer = 0.0
                    
                # 2. Add external heat (empirical split)
                q_liquid = q_net * 0.3 - (q_transfer * dt)
                q_solid = q_net * 0.7 + (q_transfer * dt)
                
                state.liquid_steel_temp_k += q_liquid / max(cap_liq, 1e-9)
                state.solid_scrap_temp_k += q_solid / max(cap_sol, 1e-9)
                
                # 3. Melting & Enthalpy mixing
                if state.solid_scrap_temp_k > cfg.steel_melt_temp_k:
                    region = "phase_change"
                    excess_j = (state.solid_scrap_temp_k - cfg.steel_melt_temp_k) * cap_sol
                    state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                    
                    melt_scrap = min(state.solid_scrap_kg, excess_j / latent)
                    q_melt = melt_scrap * latent
                    melt_rate_kg_s = melt_scrap / dt
                    
                    if melt_scrap > 0:
                        new_liq_mass = state.liquid_steel_kg + melt_scrap
                        state.liquid_steel_temp_k = (
                            state.liquid_steel_kg * state.liquid_steel_temp_k +
                            melt_scrap * cfg.steel_melt_temp_k
                        ) / new_liq_mass
                        state.solid_scrap_kg -= melt_scrap
                        state.liquid_steel_kg = new_liq_mass
                else:
                    region = "solid_heating"
            else:
                region = "liquid_superheat"
                state.solid_scrap_kg = 0.0
                state.solid_dri_kg = 0.0
                state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                
                eff_cap_liq = max(cap_liq, 5000.0 * cp_liquid)
                state.liquid_steel_temp_k += q_net / eff_cap_liq

            power_ratio = inputs["power_mw"] / 80.0
            oxy_ratio = inputs["oxygen_nm3_min"] / 80.0
            target_slag = state.liquid_steel_temp_k + 20.0 + 40.0 * power_ratio + 30.0 * oxy_ratio
            state.slag_temp_k += 0.08 * (target_slag - state.slag_temp_k)
            
            target_gas = cfg.ambient_temp_k + 250.0 + 1000.0 * (q_chem / max(dt, 1e-9) / 1e6 / 20.0) + 400.0 * power_ratio
            state.offgas_temp_k += 0.12 * (target_gas - state.offgas_temp_k)
            state.offgas_temp_k = clamp(state.offgas_temp_k, cfg.ambient_temp_k, cfg.max_offgas_temp_k)

            state.slag_kg += inputs["flux_kg_min"] / 60.0 * dt * cfg.flux_to_slag_factor
            decarb = min(state.steel_carbon_kg, inputs["oxygen_nm3_min"] / 60.0 * dt * cfg.decarb_kg_per_nm3_o2 * 0.7)
            state.steel_carbon_kg += inj_c - decarb
            
            state.steel_temp_k = state.liquid_steel_temp_k
            state.solid_scrap_temp_k = max(state.solid_scrap_temp_k, cfg.ambient_temp_k)
            state.liquid_steel_temp_k = max(state.liquid_steel_temp_k, cfg.ambient_temp_k)
            state.slag_temp_k = max(state.slag_temp_k, cfg.ambient_temp_k)
            
            tapped = start_or_continue_tapping(state, cfg)

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
                "tapped_kg_s": tapped / max(dt, 1e-9)
            }

        return self.run_loop(step)
