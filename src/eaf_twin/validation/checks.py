from __future__ import annotations

from eaf_twin.domain.models import FurnaceState


def validate_state_physics(state: FurnaceState, min_temp_k: float, max_temp_k: float) -> list[str]:
    warnings: list[str] = []
    for attr in ("solid_scrap_kg", "solid_dri_kg", "liquid_steel_kg", "slag_kg", "steel_carbon_kg", "feo_slag_kg"):
        value = getattr(state, attr)
        if value < -1e-6:
            warnings.append(f"Negative mass detected: {attr}={value:.3f}; clamped to 0.")
        setattr(state, attr, max(0.0, value))

    for attr in ("steel_temp_k", "slag_temp_k", "offgas_temp_k"):
        value = getattr(state, attr)
        if value < min_temp_k or value > max_temp_k + 350:
            warnings.append(f"Temperature out of range in {attr}: {value:.1f} K")
        setattr(state, attr, max(min_temp_k, min(max_temp_k + 350.0, value)))

    if not (0.0 <= state.melted_fraction <= 1.0):
        warnings.append(f"Melted fraction out of range: {state.melted_fraction:.4f}")
    return warnings


def plausibility_checks(summary: dict[str, float]) -> list[str]:
    issues: list[str] = []
    if not (350.0 <= summary["electric_kwh_per_tapped_t"] <= 700.0):
        issues.append("Specific electricity outside expected 350-700 kWh/t tapped")
    if summary["cum_tapped_kg"] < 92_000.0:
        issues.append("Cumulative tapped steel too low")
    if summary["final_slag_kg"] < 5_000.0 or summary["final_slag_kg"] > 22_000.0:
        issues.append("Final slag outside expected 5-22 t")
    return issues
