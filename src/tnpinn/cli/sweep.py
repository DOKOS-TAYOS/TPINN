from __future__ import annotations

import argparse
from pathlib import Path

from tnpinn.experiments import load_sweep_jobs, run_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or list a tnpinn sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-parallel", default=1, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    jobs = load_sweep_jobs(args.config)
    run_jobs(jobs, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
