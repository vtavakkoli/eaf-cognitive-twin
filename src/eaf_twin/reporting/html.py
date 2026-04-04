from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "Model_A_empirical": "Model A",
    "Model_B_first_principles": "Model B",
    "Model_C_enhanced_hybrid": "Model C",
}


def write_result_html(
    output_dir: Path,
    summary_rows: list[dict],
    scenarios: list[str],
    model_names: list[str],
) -> Path:
    """Create a single tabbed HTML report for all model/scenario outputs."""
    summary_df = pd.DataFrame(summary_rows)
    keep_cols = [
        "scenario",
        "model",
        "model_status",
        "tap_temp_c",
        "melted_fraction",
        "tap_steel_kg",
        "electric_kwh_t",
        "oxygen_nm3_t",
        "ng_nm3_t",
        "issues",
    ]

    def render_table(subset: pd.DataFrame, table_class: str = "summary-table") -> str:
        if subset.empty:
            return "<p>No rows available.</p>"
        available_cols = [c for c in keep_cols if c in subset.columns]
        return subset[available_cols].round(3).to_html(index=False, classes=table_class, border=0)

    def model_section(model_name: str) -> str:
        label = MODEL_LABELS.get(model_name, model_name)
        subset = summary_df[summary_df["model"] == model_name].copy()
        table_html = render_table(subset.drop(columns=["model"], errors="ignore"))

        plot_blocks = []
        for scen in scenarios:
            ts_name = f"timeseries_{scen}_{model_name}.csv"
            image_glob = sorted(output_dir.glob(f"plot_{scen}_{model_name}_*.png"))
            images_html = "".join(
                f'<figure><img src="{p.name}" alt="{p.name}" loading="lazy"/><figcaption>{p.name}</figcaption></figure>'
                for p in image_glob
            )
            plot_blocks.append(
                f"""
                <div class="scenario-block">
                  <h4>{scen}</h4>
                  <p><a href="{ts_name}">{ts_name}</a></p>
                  <div class="plot-grid">{images_html or "<p>No plots found.</p>"}</div>
                </div>
                """
            )

        return f"""
        <section id="{model_name}" class="tab-panel">
          <h2>{label} ({model_name})</h2>
          <h3>Summary table</h3>
          {table_html}
          <h3>Scenario plots</h3>
          {''.join(plot_blocks)}
        </section>
        """

    tabs = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" data-target="{m}">{MODEL_LABELS.get(m, m)}</button>'
        for i, m in enumerate(model_names)
    )
    panels = "".join(model_section(m) for m in model_names)
    overall_table = render_table(summary_df.copy())
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EAF Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .tabs {{ display: flex; gap: 8px; margin-bottom: 12px; }}
    .tab-btn {{ border: 1px solid #999; background: #f0f0f0; padding: 8px 12px; cursor: pointer; }}
    .tab-btn.active {{ background: #d7e8ff; border-color: #2c6fbb; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .summary-table {{ border-collapse: collapse; width: 100%; margin-bottom: 14px; }}
    .summary-table th, .summary-table td {{ border: 1px solid #ccc; padding: 6px; font-size: 13px; }}
    .scenario-block {{ margin: 16px 0 26px; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }}
    .plot-grid figure {{ margin: 0; }}
    .plot-grid img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    .plot-grid figcaption {{ font-size: 12px; color: #444; padding-top: 4px; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>EAF Simulation Results</h1>
  <h2>All-model summary</h2>
  {overall_table}
  <ul>
    <li><a href="result.html">result.html</a></li>
  </ul>
  <div class="tabs">{tabs}</div>
  {panels}
  <script>
    const buttons = Array.from(document.querySelectorAll('.tab-btn'));
    const panels = Array.from(document.querySelectorAll('.tab-panel'));
    function activate(target) {{
      buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.target === target));
      panels.forEach(panel => panel.classList.toggle('active', panel.id === target));
    }}
    buttons.forEach(btn => btn.addEventListener('click', () => activate(btn.dataset.target)));
    if (buttons.length > 0) activate(buttons[0].dataset.target);
  </script>
</body>
</html>
"""
    out_path = output_dir / "result.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
