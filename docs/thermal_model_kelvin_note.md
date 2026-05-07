# Thermal model correction note (Model B / Model C)

## What was wrong

The first-principles and enhanced-hybrid thermal calculations mixed Celsius and Kelvin in internal state updates. In addition, the default initial steel/slag temperatures started at near-liquid conditions, which produced physically inconsistent trajectories (hot-start behavior rather than cold charge melting).

## What was corrected

1. **Internal thermal state moved to Kelvin**
   - `steel_temp_k`, `slag_temp_k`, and `offgas_temp_k` are now the internal state variables.
   - Configuration keeps Celsius inputs for convenience, with explicit `*_temp_k` conversion properties.
   - Radiation (`T^4`), sensible heating, and thermal losses all use Kelvin internally.

2. **Cold-charge initial conditions**
   - Ambient and scrap defaults now use **23 °C = 296.15 K**.
   - Initial steel/slag/off-gas defaults are cold-charge values near ambient.
   - Optional hot heel is modeled explicitly via `initial_hot_heel_kg` and `initial_hot_heel_temp_c`.

3. **Enthalpy-driven melt and temperature evolution**
   - Added explicit helper functions for:
     - steel solid sensible heat
     - steel latent heat
     - steel liquid sensible heat
     - slag sensible enthalpy
     - off-gas sensible enthalpy
   - Available net enthalpy is applied in physically ordered steps:
     1) heat metallic charge to melt range,
     2) consume latent heat to melt solids,
     3) superheat liquid steel after melting.

4. **Dynamic slag and off-gas thermal balances**
   - Slag temperature is solved from an energy balance with arc/burner/reaction partitioning, slag-metal exchange, and losses.
   - Off-gas temperature is dynamic and bounded using Kelvin limits.

5. **State guards and tap logic improvements**
   - Validation now clamps/flags Kelvin temperatures and checks melted-fraction bounds.
   - Tap readiness now requires high melted fraction and steel tap temperature in Kelvin.

## Why the new behavior is more realistic

The revised model prevents nonphysical combinations such as full melt at low bath temperature and removes artificial hot-start behavior. Thermal evolution now emerges from net enthalpy input and transfer/loss terms, producing a cold-charge rise toward melting and then superheat/tap conditions.
