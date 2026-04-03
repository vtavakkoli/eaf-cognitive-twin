from __future__ import annotations

import argparse
from pathlib import Path

from eaf_twin.config.loader import load_config
from eaf_twin.simulation.runner import run_full_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EAF cognitive digital twin")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--dt", type=float, default=None)
    run.add_argument("--noise", type=float, default=None)
    run.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        cfg = load_config(args.config)
        if args.dt is not None:
            cfg.dt_s = args.dt
        if args.noise is not None:
            cfg.measurement_noise_std = args.noise
        if args.seed is not None:
            cfg.random_seed = args.seed
        df = run_full_simulation(cfg, args.output_dir)
        print(df[["scenario", "model", "cum_tapped_kg", "electric_kwh_per_tapped_t", "model_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
