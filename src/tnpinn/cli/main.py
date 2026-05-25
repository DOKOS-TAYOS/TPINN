from __future__ import annotations

import argparse

from tnpinn.cli import analyze, benchmark, evaluate, plot, sanity_check, sweep, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tnpinn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="train one experiment")
    subparsers.add_parser("sweep", help="run or list a sweep")
    subparsers.add_parser("benchmark", help="run or list a benchmark suite")
    subparsers.add_parser("evaluate", help="evaluate one run")
    subparsers.add_parser("analyze", help="aggregate runs")
    subparsers.add_parser("plot", help="regenerate run figures")
    subparsers.add_parser("sanity-check", help="run a short end-to-end sanity suite")
    args, rest = parser.parse_known_args(argv)
    dispatch = {
        "train": train.main,
        "sweep": sweep.main,
        "benchmark": benchmark.main,
        "evaluate": evaluate.main,
        "analyze": analyze.main,
        "plot": plot.main,
        "sanity-check": sanity_check.main,
    }
    dispatch[args.command](rest)


if __name__ == "__main__":
    main()
