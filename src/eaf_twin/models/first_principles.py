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
        if enhanced: self.name = "Model_C_enhanced_hybrid"

    def simulate(self):
        cfg = self.config

        def step(state, inputs, warnings):
            dt = cfg.dt_s
            stg = stage_name(state.time_s / SECONDS_PER_MIN, state.melted_fraction)
            
            # Energy Flow
            q_useful = (inputs["power_mw"] * 1e6 * cfg.eta_arc_melting) * dt
            q_chem = (inputs["ng_nm3_min"]/60 * cfg.lhv_ng_j_nm3 + inputs["oxygen_nm3_min"]/60 * cfg.oxygen_heat_j_nm3) * dt

            # Slag hotter than steel (receives more direct arc/chem energy)
            q_to_slag = (q_useful * 0.20 + q_chem * 0.35)
            q_to_metal = (q_useful + q_chem) - q_to_slag
            
            # Heat Loss (Reduced UA effect to allow simulation to finish melting)
            t_int_k = 0.7 * state.liquid_steel_temp_k + 0.3 * state.slag_temp_k
            q_loss = (cfg.ua_wall_w_k * 0.4) * (t_int_k - cfg.ambient_temp_k) * dt

            # Update Slag
            state.slag_temp_k += (q_to_slag - 0.2 * q_loss) / max(state.slag_kg * cfg.cp_slag_j_kgk, 1e-9)
            
            f_melt = state.melted_fraction
            q_liquid = q_to_metal * (f_melt**2) - 0.5 * q_loss
            q_solid = q_to_metal - (q_to_metal * (f_melt**2)) - 0.1 * q_loss

            solid_mass = state.solid_scrap_kg + state.solid_dri_kg
            if solid_mass > 1e-6:
                # Convective mixing (bath helps melt scrap)
                q_mix = min(q_liquid, 40000.0 * (state.liquid_steel_temp_k - state.solid_scrap_temp_k) * dt)
                state.solid_scrap_temp_k += (q_solid + q_mix) / (solid_mass * cfg.cp_scrap_j_kgk)
                state.liquid_steel_temp_k += (q_liquid - q_mix) / (state.liquid_steel_kg * cfg.cp_steel_j_kgk)
                
                if state.solid_scrap_temp_k >= cfg.steel_melt_temp_k:
                    state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                    melt_j = q_solid + q_mix
                    # If bath is superheated, dump that energy into melting too
                    if state.liquid_steel_temp_k > cfg.steel_melt_temp_k:
                        ex_j = (state.liquid_steel_temp_k - cfg.steel_melt_temp_k) * state.liquid_steel_kg * cfg.cp_steel_j_kgk * 0.8
                        melt_j += ex_j
                        state.liquid_steel_temp_k -= ex_j / (state.liquid_steel_kg * cfg.cp_steel_j_kgk)
                    
                    melt_kg = min(state.solid_scrap_kg, melt_j / cfg.latent_heat_steel_j_kg)
                    if melt_kg > 0:
                        total_liq = state.liquid_steel_kg + melt_kg
                        state.liquid_steel_temp_k = (state.liquid_steel_kg * state.liquid_steel_temp_k + melt_kg * cfg.steel_melt_temp_k) / total_liq
                        state.solid_scrap_kg -= melt_kg
                        state.liquid_steel_kg = total_liq
            else:
                # Superheat only after solid is gone
                state.solid_scrap_temp_k = cfg.steel_melt_temp_k
                eff_cap = max(state.liquid_steel_kg * cfg.cp_steel_j_kgk, 5000.0 * cfg.cp_steel_j_kgk)
                state.liquid_steel_temp_k += q_liquid / eff_cap

            state.offgas_temp_k = clamp(cfg.ambient_temp_k + 400.0 + 800.0 * (inputs["power_mw"]/80), cfg.ambient_temp_k, cfg.max_offgas_temp_k)
            state.steel_temp_k = state.liquid_steel_temp_k
            tapped = start_or_continue_tapping(state, cfg)

            state.cum_electric_j += (inputs["power_mw"] * 1e6 * dt)
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += (q_useful + q_chem)
            state.cum_losses_j += q_loss
            state.cum_oxygen_nm3 += inputs["oxygen_nm3_min"]/60*dt
            state.cum_ng_nm3 += inputs["ng_nm3_min"]/60*dt
            state.cum_carbon_kg += inputs["carbon_kg_min"]/60*dt
            
            return {"stage": stg, "q_useful_mw": (q_useful+q_chem)/dt/1e6, "q_melt_mw": melt_kg*cfg.latent_heat_steel_j_kg/dt/1e6 if solid_mass > 0 else 0, "q_loss_mw": q_loss/dt/1e6, "tapped_kg_s": tapped/dt}

        return self.run_loop(step)
