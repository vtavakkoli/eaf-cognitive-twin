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
            
            # 1. Base Inputs
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

            # 2. Chemistry & Mass Balances
            fe_oxid = min(state.liquid_steel_kg * 0.0015, o2_flow * cfg.fe_oxidation_ratio_per_nm3_o2 * dt)
            oxide = 1.29 * fe_oxid
            decarb = min(state.steel_carbon_kg, o2_flow * cfg.decarb_kg_per_nm3_o2 * dt)
            state.steel_carbon_kg += c_flow * dt - decarb - 0.0006 * state.steel_carbon_kg * dt
            state.feo_slag_kg += oxide
            state.liquid_steel_kg = max(0.0, state.liquid_steel_kg - fe_oxid)

            solid_mass = max(state.solid_scrap_kg + state.solid_dri_kg, 0.0)
            liquid_mass = max(state.liquid_steel_kg, 0.0)
            total_metal_mass = max(liquid_mass + solid_mass, 1.0)
            sf = solid_mass / total_metal_mass

            # Calculate specific heat capacities early for thermal constraints
            cp_sol = cp_solid_steel_j_kgk(state.solid_scrap_temp_k)
            cp_liq = cp_liquid_steel_j_kgk(state.liquid_steel_temp_k)

            # 3. Energy Routing (5 Medium Framework)
            # Solid Scrap directly absorbs massive heat from the arc and burners
            q_arc_ss = q_arc_useful * sf * 0.85
            q_arc_mm = q_arc_useful * (1.0 - sf) * 0.85
            q_arc_slag = q_arc_useful * 0.15

            q_burn_ss = q_burn * sf * 0.70
            q_burn_mm = q_burn * (1.0 - sf) * 0.70
            q_burn_slag = q_burn * 0.30

            # Chemical reactions occur inside the liquid bath/slag
            q_chem_mm = (q_oxy + q_c) * 0.80
            q_chem_slag = (q_oxy + q_c) * 0.20

            # 4. Thermal Coupling
            k_mm_ss = 30000.0 + 20000.0 * (liquid_mass / 100000.0)
            q_mm_to_ss_unbounded = k_mm_ss * max(0.0, state.liquid_steel_temp_k - state.solid_scrap_temp_k) * dt
            
            # Prevent liquid steel from cooling significantly below its freezing point without undergoing tracking phase change back to solid
            max_q_transfer = max(0.0, liquid_mass * cp_liq * (state.liquid_steel_temp_k - (t_m - 30.0)))
            q_mm_to_ss = min(q_mm_to_ss_unbounded, max_q_transfer)
            q_ss_to_mm = k_mm_ss * max(0.0, state.solid_scrap_temp_k - state.liquid_steel_temp_k) * dt

            # Bath to Slag coupling prevents slag from artificially decoupling
            k_mm_slag = cfg.slag_to_bath_heat_coeff_w_k
            q_mm_to_slag = k_mm_slag * (state.liquid_steel_temp_k - state.slag_temp_k) * dt

            # 5. Heat Losses
            # Only hot liquid and slag radiate. Solid scrap is cold, so it does not lose heat to the walls!
            t_int_k = 0.6 * state.liquid_steel_temp_k + 0.4 * state.slag_temp_k
            q_wall = cfg.ua_wall_w_k * max(0.0, t_int_k - cfg.ambient_temp_k) * dt * 0.35
            q_rad = cfg.radiation_loss_factor * (1.0 - 0.3 * foam) * SIGMA * cfg.area_effective_m2 * max(0.0, t_int_k**4 - cfg.ambient_temp_k**4) * dt * 0.35
            
            q_loss_total = q_wall + q_rad
            q_loss_mm = q_loss_total * 0.40
            q_loss_slag = q_loss_total * 0.60

            # Solid Flux (Implicit 5th medium: absorbs energy to heat up and melt into slag)
            q_flux_sink = flux_flow * dt * (cfg.cp_slag_j_kgk * max(0.0, state.slag_temp_k - cfg.ambient_temp_k) + 200000.0)

            # Net Energies for the 3 main tracked mediums
            q_net_ss = q_arc_ss + q_burn_ss + q_mm_to_ss - q_ss_to_mm
            q_net_mm = q_arc_mm + q_burn_mm + q_chem_mm - q_mm_to_ss + q_ss_to_mm - q_loss_mm - q_mm_to_slag
            q_net_slag = q_arc_slag + q_burn_slag + q_chem_slag - q_loss_slag - q_flux_sink + q_mm_to_slag

            # 6. Apply Heat & Phase Changes
            melt_rate = 0.0
            region = "liquid_superheat"
            q_melt_actual = 0.0

            # Heat Solid Scrap
            if solid_mass > 1e-6:
                region = "solid_heating"
                cap_ss = solid_mass * cp_sol
                state.solid_scrap_temp_k += q_net_ss / max(cap_ss, 1e-9)

                # Melting threshold hit
                if state.solid_scrap_temp_k > t_m:
                    region = "phase_change"
                    excess_j = (state.solid_scrap_temp_k - t_m) * cap_ss
                    state.solid_scrap_temp_k = t_m

                    latent_scrap = cfg.latent_heat_steel_j_kg
                    latent_dri = cfg.latent_heat_steel_j_kg + cfg.dri_reduction_endotherm_j_kg
                    latent_mix = (state.solid_scrap_kg * latent_scrap + state.solid_dri_kg * latent_dri) / max(solid_mass, 1e-9)
                    
                    melt_kg = excess_j / max(latent_mix, 1e-9)
                    
                    if melt_kg >= solid_mass:
                        melt_kg = solid_mass
                        q_net_mm += (excess_j - melt_kg * latent_mix)

                    scrap_ratio = state.solid_scrap_kg / solid_mass
                    dri_ratio = state.solid_dri_kg / solid_mass
                    m_scrap = melt_kg * scrap_ratio
                    m_dri = melt_kg * dri_ratio

                    state.solid_scrap_kg -= m_scrap
                    state.solid_dri_kg -= m_dri
                    state.liquid_steel_kg += m_scrap + m_dri * cfg.dri_fe_metallization
                    state.slag_kg += m_dri * (1.0 - cfg.dri_fe_metallization)
                    
                    q_melt_actual = m_scrap * latent_scrap + m_dri * latent_dri
                    state.cum_latent_heat_j += q_melt_actual
                    melt_rate = melt_kg / max(dt, 1e-9)
            else:
                q_net_mm += q_net_ss
                state.solid_scrap_temp_k = state.liquid_steel_temp_k

            # Update Liquid Steel
            if state.liquid_steel_kg > 1e-6:
                cap_mm = state.liquid_steel_kg * cp_liq
                state.liquid_steel_temp_k += q_net_mm / max(cap_mm, 1e-9)
            else:
                state.liquid_steel_temp_k = state.solid_scrap_temp_k

            # Update Liquid Slag
            state.slag_kg += flux_flow * dt * cfg.flux_to_slag_factor + oxide
            if state.slag_kg > 1e-6:
                cap_slag = state.slag_kg * cfg.cp_slag_j_kgk
                state.slag_temp_k += q_net_slag / max(cap_slag, 1e-9)
            else:
                state.slag_temp_k = state.liquid_steel_temp_k

            # Update Off-gas (5th Medium)
            target_gas = cfg.ambient_temp_k + 400.0 + (power_w / 80e6) * 1000.0 + (q_chem / max(dt, 1e-9) / 20e6) * 500.0
            if state.roof_open_remaining_s > 0:
                target_gas = cfg.ambient_temp_k + 50.0
            state.offgas_temp_k += (target_gas - state.offgas_temp_k) * clamp(dt / 30.0, 0.05, 1.0)

            # Enforce physical bounds
            state.steel_temp_k = state.liquid_steel_temp_k
            state.solid_scrap_temp_k = max(state.solid_scrap_temp_k, cfg.ambient_temp_k)
            state.liquid_steel_temp_k = max(state.liquid_steel_temp_k, cfg.ambient_temp_k)
            state.slag_temp_k = max(state.slag_temp_k, cfg.ambient_temp_k)
            state.offgas_temp_k = clamp(state.offgas_temp_k, cfg.ambient_temp_k, cfg.max_offgas_temp_k)

            tapped = start_or_continue_tapping(state, cfg)

            # Trackers
            state.cum_electric_j += q_elec
            state.cum_chemical_j += q_chem
            state.cum_useful_heat_j += q_arc_useful + q_burn + q_oxy + q_c
            state.cum_losses_j += q_loss_total
            state.cum_oxygen_nm3 += o2_flow * dt
            state.cum_ng_nm3 += ng_flow * dt
            state.cum_carbon_kg += c_flow * dt

            return {
                "stage": stg,
                "foamy_factor": foam,
                "eta_arc": eta,
                "q_useful_mw": (q_arc_useful + q_burn + q_oxy + q_c) / max(dt, 1e-9) / 1e6,
                "q_melt_mw": q_melt_actual / max(dt, 1e-9) / 1e6,
                "q_loss_mw": q_loss_total / max(dt, 1e-9) / 1e6,
                "melt_rate_kg_s": melt_rate,
                "phase_region": region,
                "remaining_solid_kg": state.solid_scrap_kg + state.solid_dri_kg,
                "latent_heat_consumed_gj": state.cum_latent_heat_j / 1e9,
                "enthalpy_balance_residual_mj": 0.0,
                "tapped_kg_s": tapped / max(dt, 1e-9),
            }

        return self.run_loop(step)
