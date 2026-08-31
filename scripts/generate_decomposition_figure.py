"""Render the mode-wise amplitude/phase error decomposition.

Splits the endpoint-field squared vorticity error at each requested step into
the part caused by getting each Fourier mode's magnitude wrong and the part
caused by getting its phase wrong, paralleling the mode-wise magnitude/phase
error reading in Köhler, "From Numerical Simulators of PDEs to Neural
Emulators and Back" (2026), arXiv:2608.24547 (closed-form there, for Fourier
multipliers of linear schemes; the identity used here is its empirical
counterpart for a nonlinear system).

Diagnostic only: validation split, and no reported result depends on it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "docs" / "results" / "spectra-validation.json"
DEFAULT_FIGURE = REPO_ROOT / "docs" / "figures" / "fig_error_decomposition"

METHOD_STYLE: dict[str, dict] = {
    "aposteriori": {
        "label": "A-posteriori (selected)",
        "colour": "#0072B2",
        "linestyle": "-",
    },
    "apriori": {
        "label": "A-priori (matched)",
        "colour": "#D55E00",
        "linestyle": "--",
    },
    "no-closure": {
        "label": "No closure",
        "colour": "#009E73",
        "linestyle": "-.",
    },
    "dynamic-smagorinsky": {
        "label": "Dynamic Smagorinsky",
        "colour": "#E69F00",
        "linestyle": ":",
    },
    "static-smagorinsky": {
        "label": "Static Smagorinsky",
        "colour": "#6E6E6E",
        "linestyle": (0, (5, 2)),
    },
}

RC_PARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "axes.grid": False,
    "svg.hashsalt": "tesseract-hybrid-closure-decomposition",
}


def _normalise_svg(path: Path) -> None:
    """Strip the matplotlib comment and trailing whitespace for byte stability.

    The SVG backend emits trailing spaces on long path lines; stripping them
    makes the output stable and clean for ``git diff --check``.
    """
    text = path.read_text(encoding="utf-8")
    lines = [
        line.rstrip(" \t")
        for line in text.splitlines()
        if "Created with matplotlib" not in line
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_report(path: Path) -> Mapping:
    if not path.is_file():
        raise SystemExit(
            f"report not found: {path}\nGenerate it first with 'make spectra'."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if "error_decomposition" not in report:
        raise SystemExit(
            "report predates the error decomposition; regenerate with 'make spectra'"
        )
    return report


def render(report: Mapping, base: Path, steps: Sequence[int], max_wavenumber: int):
    """Per-shell error above, the amplitude/phase split of its total below.

    The per-shell phase *share* turned out to sit near one half for every
    method at every scale, so it discriminates poorly; the aggregate split
    does not, and stacked totals show it directly while keeping the bar
    height equal to that step's endpoint squared error.
    """
    wavenumber = np.asarray(report["decomposition_wavenumber"], dtype=float)
    keep = (wavenumber >= 1) & (wavenumber <= max_wavenumber)
    available = report["error_decomposition"]
    missing = [step for step in steps if str(step) not in available]
    if missing:
        raise SystemExit(f"steps {missing} not in report")

    with plt.rc_context(RC_PARAMS):
        figure, axes = plt.subplots(
            2,
            len(steps),
            figsize=(3.7 * len(steps) + 0.8, 6.6),
            layout="constrained",
        )
        for column, step in enumerate(steps):
            entries = available[str(step)]
            top, bottom = axes[0][column], axes[1][column]

            names = [name for name in METHOD_STYLE if name in entries]
            for name in names:
                style = METHOD_STYLE[name]
                amplitude = np.asarray(entries[name]["amplitude"], dtype=float)[keep]
                phase = np.asarray(entries[name]["phase"], dtype=float)[keep]
                top.loglog(
                    wavenumber[keep],
                    amplitude + phase,
                    color=style["colour"],
                    linestyle=style["linestyle"],
                    linewidth=1.6,
                    label=style["label"],
                )
            top.set_title(f"step {step}")
            top.set_xlabel("wavenumber $k$")

            positions = np.arange(len(names))
            amplitudes = [entries[name]["amplitude_total"] for name in names]
            phases = [entries[name]["phase_total"] for name in names]
            colours = [METHOD_STYLE[name]["colour"] for name in names]
            bottom.bar(
                positions,
                amplitudes,
                color=colours,
                width=0.62,
                label="amplitude",
            )
            bottom.bar(
                positions,
                phases,
                bottom=amplitudes,
                color=colours,
                width=0.62,
                alpha=0.45,
                hatch="///",
                label="phase",
            )
            bottom.set_xticks(positions)
            bottom.set_xticklabels(
                [METHOD_STYLE[name]["label"].split(" (")[0] for name in names],
                rotation=35,
                ha="right",
            )
            bottom.set_title(f"step {step}")

        axes[0][0].set_ylabel("squared vorticity error per shell")
        axes[1][0].set_ylabel("total squared vorticity error")

        handles, labels = axes[0][0].get_legend_handles_labels()
        split_handles, split_labels = axes[1][0].get_legend_handles_labels()
        figure.legend(
            handles + split_handles[:2],
            labels + ["amplitude part", "phase part (hatched)"],
            loc="outside lower center",
            ncols=4,
            frameon=False,
        )
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
    parser.add_argument("--steps", nargs="+", type=int, default=[30, 120, 500])
    parser.add_argument("--max-wavenumber", type=int, default=32)
    args = parser.parse_args(argv)

    report = load_report(args.report)
    written = render(report, args.output_base, args.steps, args.max_wavenumber)
    print(f"wrote {written}")
    summary = {
        step: {
            name: {
                "phase_fraction": round(entry["phase_fraction"], 4),
                "squared_error_total": entry["squared_error_total"],
            }
            for name, entry in report["error_decomposition"][step].items()
        }
        for step in report["error_decomposition"]
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
