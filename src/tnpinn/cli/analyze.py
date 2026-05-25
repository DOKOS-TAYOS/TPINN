from __future__ import annotations

import argparse
from pathlib import Path

from tnpinn.results.analysis import analyze_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate tnpinn run results.")
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = analyze_runs(args.runs, args.out)
    print(f"wrote {len(summary)} rows to {args.out}")


if __name__ == "__main__":
    main()
