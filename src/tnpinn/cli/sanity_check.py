from __future__ import annotations

import argparse
from pathlib import Path

from tnpinn.sanity import SanityOptions, run_sanity_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short end-to-end tnpinn sanity suite.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--function-steps", default=500, type=int)
    parser.add_argument("--laplace-steps", default=250, type=int)
    parser.add_argument("--hard-steps", default=50, type=int)
    parser.add_argument("--benchmark-steps", default=None, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_sanity_checks(
        args.out,
        SanityOptions(
            function_steps=args.function_steps,
            laplace_steps=args.laplace_steps,
            hard_steps=args.hard_steps,
            benchmark_steps=args.benchmark_steps,
            seed=args.seed,
        ),
    )
    print(f"wrote sanity report to {args.out}")
    if not bool(summary.get("passed", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
