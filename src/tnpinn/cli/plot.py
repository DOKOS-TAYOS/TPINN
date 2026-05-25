from __future__ import annotations

import argparse
from pathlib import Path

from tnpinn.viz.heat import plot_heat_prediction
from tnpinn.viz.training import plot_training_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate figures for one tnpinn run.")
    parser.add_argument("--run", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    figures_dir = args.run / "figures"
    plot_training_curves(args.run / "metrics.csv", figures_dir)
    plot_heat_prediction(args.run / "predictions.npz", figures_dir)
    print(figures_dir)


if __name__ == "__main__":
    main()
