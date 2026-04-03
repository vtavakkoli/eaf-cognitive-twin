# Assumptions

## What is modeled
- Lumped dynamic mass/energy balances for steel, slag, and off-gas.
- Time-varying operational stages (power, oxygen, fuel, carbon, flux).
- Scrap and DRI charging events with thermal shock.
- Fe/C/FeO and flux-to-slag simplified metallurgy.
- Explicit tapping state and cumulative tapped steel KPI.

## What is simplified
- Single-zone temperature for steel, slag, and off-gas.
- Reduced chemistry (no full equilibrium calculation).
- Simple foamy-slag influence in enhanced model.
- Deterministic operation unless explicit sensor noise is enabled.

## Not yet modeled
- Detailed Si/Mn/P/S kinetics.
- Electrode wear and arc impedance physics.
- CFD-level bath mixing and spatial gradients.
- Plant historian data assimilation and online state estimation.
