def _apply_metal_charge_event(
        self,
        state: FurnaceState,
        scrap_kg: float,
        dri_kg: float,
        charge_temp_k: float,
        interaction_factor: float = 1.0,  # Ensure full thermal mixing to match target plots
    ) -> None:
        if scrap_kg <= 0 and dri_kg <= 0:
            return

        added_mass = scrap_kg + dri_kg
        old_solid = max(state.solid_scrap_kg + state.solid_dri_kg - added_mass, 0.0)
        
        if old_solid > 0:
            state.solid_scrap_temp_k = (
                old_solid * state.solid_scrap_temp_k + added_mass * charge_temp_k
            ) / max(old_solid + added_mass, 1e-9)
        else:
            state.solid_scrap_temp_k = charge_temp_k

        m_liq = max(state.liquid_steel_kg, 0.0)
        if m_liq > 1e-6:
            # Enforce true thermodynamic mixing (no artificial superheat limits)
            interacting_mass = interaction_factor * added_mass
            cap_liq = m_liq * self.config.cp_steel_j_kgk
            cap_sol = interacting_mass * self.config.cp_scrap_j_kgk
            
            mixed_temp = (cap_liq * state.liquid_steel_temp_k + cap_sol * charge_temp_k) / max(cap_liq + cap_sol, 1e-9)
            state.liquid_steel_temp_k = mixed_temp
            
        state.steel_temp_k = state.liquid_steel_temp_k
