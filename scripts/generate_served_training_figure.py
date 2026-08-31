"""Render the served-container training run as a loss curve.

Reads the report written by ``hybrid-closure train-served`` and plots the
per-update objective, whose every gradient crossed the served Tesseract
boundary, alongside a running mean and the held-out validation comparison.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "docs" / "results" / "served-training.json"
DEFAULT_FIGURE = REPO_ROOT / "docs" / "figures" / "fig_served_training"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GRAPHITE = "#000000"
GREY = "#6E6E6E"

RC_PARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.grid": False,
    "svg.hashsalt": "tesseract-hybrid-closure-served-training",
}


def _normalise_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = [
        line.rstrip(" \t")
        for line in text.splitlines()
        if "Created with matplotlib" not in line
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _running_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_report(path: Path) -> Mapping:
    if not path.is_file():
        raise SystemExit(
            f"served training report not found: {path}\n"
            "Generate it first with 'make served-training'."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("use_images"):
        raise SystemExit(
            "refusing to plot a run that did not use the served container images"
        )
    return report


def render(report: Mapping, base: Path) -> Path:
    losses = np.asarray(report["loss_curve"], dtype=float)
    updates = np.arange(1, losses.size + 1)
    window = max(1, losses.size // 25)

    with plt.rc_context(RC_PARAMS):
        figure, (left, right) = plt.subplots(
            1, 2, figsize=(10.4, 4.0), layout="constrained"
        )

        left.semilogy(
            updates,
            losses,
            color=BLUE,
            linewidth=0.7,
            alpha=0.35,
            label="per-update objective",
        )
        smoothed = _running_mean(losses, window)
        left.semilogy(
            updates[window - 1 :],
            smoothed,
            color=BLUE,
            linewidth=2.0,
            label=f"running mean ({window} updates)",
        )
        left.set_xlabel("Adam update (one train-split trajectory each)")
        left.set_ylabel(f"{report['unroll_steps']}-step rollout vorticity MSE")
        left.text(-0.10, 1.02, "(a)", transform=left.transAxes, fontweight="bold")
        left.legend(frameon=False)

        labels = ["Untrained\ninitialisation", "Served-trained\n(this run)"]
        values = [
            report["validation_mse_before"],
            report["validation_mse_after"],
        ]
        colours = [GREY, BLUE]
        reference = report.get("in_process_reference_mse")
        if reference is not None:
            labels.append("In-process\nreported model")
            values.append(reference)
            colours.append(VERMILLION)

        bars = right.bar(labels, values, color=colours, width=0.6)
        right.set_yscale("log")
        right.set_ylabel(f"{report['validation_unroll']}-step validation vorticity MSE")
        right.text(-0.10, 1.02, "(b)", transform=right.transAxes, fontweight="bold")
        for bar, value in zip(bars, values, strict=True):
            right.annotate(
                f"{value:.5f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=8.5,
            )
        right.margins(y=0.25)

        base.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("svg", "png"):
            destination = base.with_suffix(f".{suffix}")
            figure.savefig(
                destination,
                bbox_inches="tight",
                pad_inches=0.03,
                metadata={"Date": None} if suffix == "svg" else {},
            )
            if suffix == "svg":
                _normalise_svg(destination)
        plt.close(figure)
    return base.with_suffix(".svg")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args(argv)

    report = load_report(args.report)
    written = render(report, args.output_base)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
