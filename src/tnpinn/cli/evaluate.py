from __future__ import annotations

import argparse
from pathlib import Path

from tnpinn.pinn.evaluator import evaluate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained tnpinn run.")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--grid-size", default=None, type=int)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_run(
        args.run,
        checkpoint=args.checkpoint,
        grid_size=args.grid_size,
        save_predictions=args.save_predictions,
        make_figures=args.make_figures,
    )


if __name__ == "__main__":
    main()
