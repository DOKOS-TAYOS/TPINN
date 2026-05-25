from __future__ import annotations

import argparse
from pathlib import Path

from tnpinn.config.loading import load_config
from tnpinn.pinn.trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one tnpinn experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", default=None, type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config, overrides=args.overrides)
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    run_dir = train(config, run_id=args.run_id, dry_run=args.dry_run)
    print(run_dir)


if __name__ == "__main__":
    main()
