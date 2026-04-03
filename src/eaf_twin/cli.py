from __future__ import annotations

import argparse
from pathlib import Path

import eaf_simulator as legacy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EAF cognitive digital twin")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run full legacy-calibrated simulation pipeline")
    run.add_argument("--config", type=Path, default=None)
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--dt", type=float, default=None)
    run.add_argument("--noise", type=float, default=None)
    run.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "run":
        raise ValueError(f"Unsupported command: {args.command}")

    cfg = legacy.load_config(args.config)
    if args.dt is not None:
        if args.dt <= 0 or args.dt > 20:
            raise ValueError("dt must be within (0, 20] seconds")
        cfg.dt_s = args.dt
    if args.noise is not None:
        if args.noise < 0 or args.noise > 30:
            raise ValueError("noise must be within [0, 30] °C")
        cfg.measurement_noise_std = args.noise
    if args.seed is not None:
        cfg.random_seed = args.seed

    legacy.run_full_simulation(cfg, args.output_dir)


if __name__ == "__main__":
    main()
