"""Plot backtest equity curves."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot equity curve from backtest results")
    parser.add_argument("--equity", required=True, help="NPY or CSV file containing equity curve data")
    parser.add_argument("--output", help="Optional path to save plot instead of showing")
    parser.add_argument("--title", default="Equity Curve", help="Plot title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    equity = _load_equity(args.equity)
    if equity.size == 0:
        raise ValueError("Equity curve is empty")
    plt.figure(figsize=(10, 4))
    plt.plot(equity)
    plt.title(args.title)
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.grid(True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, bbox_inches="tight")
        print(f"Saved plot to {args.output}")
    else:  # pragma: no cover - interactive use
        plt.show()


def _load_equity(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".csv"):
        frame = pl.read_csv(path)
        column = frame.columns[0]
        return frame[column].to_numpy()
    raise ValueError("Unsupported equity file format; use .npy or .csv")


if __name__ == "__main__":
    main()
