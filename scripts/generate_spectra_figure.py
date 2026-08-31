"""Render the validation-split spectral diagnostic figure.

Reads the report written by ``hybrid-closure spectra`` and produces a
four-panel figure: energy and enstrophy spectra against the filtered-DNS
reference, and each method's ratio to that reference, which is where the
differences between methods are actually legible.

Diagnostic only. This script never reads the sealed test split and regenerates
none of the reported result assets.
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
DEFAULT_FIGURE = REPO_ROOT / "docs" / "figures" / "fig_spectra"

OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "graphite": "#000000",
}

METHOD_STYLE: dict[str, dict] = {
    "filtered-dns": {
        "label": "Filtered DNS",
        "colour": OKABE_ITO["graphite"],
        "linestyle": "-",
        "linewidth": 2.2,
        "zorder": 5,
    },
    "aposteriori": {
        "label": "A-posteriori (selected)",
        "colour": OKABE_ITO["blue"],
        "linestyle": "-",
        "linewidth": 1.6,
        "zorder": 4,
    },
    "apriori": {
        "label": "A-priori (matched)",
        "colour": OKABE_ITO["vermillion"],
        "linestyle": "--",
        "linewidth": 1.6,
        "zorder": 3,
    },
    "no-closure": {
        "label": "No closure",
        "colour": OKABE_ITO["bluish_green"],
        "linestyle": "-.",
        "linewidth": 1.6,
        "zorder": 2,
    },
    "dynamic-smagorinsky": {
        "label": "Dynamic Smagorinsky",
        "colour": OKABE_ITO["orange"],
        "linestyle": ":",
        "linewidth": 1.4,
        "zorder": 1,
    },
    "static-smagorinsky": {
        "label": "Static Smagorinsky",
        "colour": "#6E6E6E",
        "linestyle": (0, (5, 2)),
        "linewidth": 1.4,
        "zorder": 1,
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
    "svg.hashsalt": "tesseract-hybrid-closure-spectra",
}


def _normalise_svg(path: Path) -> None:
    """Strip generated comments and trailing whitespace for byte stability."""
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
            f"spectral report not found: {path}\nGenerate it first with 'make spectra'."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("split") != "validation":
        raise SystemExit(
            "refusing to plot a report that is not from the validation split"
        )
    return report


def _plot_spectra(axis, wavenumber, spectra: Mapping, quantity: str, step: int) -> None:
    for name, style in METHOD_STYLE.items():
        if name not in spectra:
            continue
        values = np.asarray(spectra[name][quantity], dtype=float)
        axis.loglog(
            wavenumber,
            values,
            color=style["colour"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=style["zorder"],
            label=style["label"],
        )
    symbol = "E(k)" if quantity == "energy" else "Z(k)"
    axis.set_ylabel(symbol)
    axis.set_title(f"step {step}")


def _plot_ratios(axis, wavenumber, spectra: Mapping, quantity: str, step: int) -> None:
    reference = np.asarray(spectra["filtered-dns"][quantity], dtype=float)
    for name, style in METHOD_STYLE.items():
        if name == "filtered-dns" or name not in spectra:
            continue
        values = np.asarray(spectra[name][quantity], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(reference > 0.0, values / reference, np.nan)
        axis.semilogx(
            wavenumber,
            ratio,
            color=style["colour"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=style["zorder"],
            label=style["label"],
        )
    axis.axhline(1.0, color=OKABE_ITO["graphite"], linewidth=1.4, zorder=5)
    symbol = "E(k)" if quantity == "energy" else "Z(k)"
    axis.set_xlabel("wavenumber $k$")
    axis.set_ylabel(f"{symbol} / {symbol}$_{{\\mathrm{{DNS}}}}$")


def render(report: Mapping, base: Path, steps: Sequence[int]) -> Path:
    """Energy spectra on top, enstrophy ratio to DNS below, one column per step.

    The enstrophy ratio is the lower row because the training objective is a
    vorticity MSE. Energy and enstrophy differ by the exact squared wavenumber
    factor mode by mode, and their shell-ratio rows are empirically close for
    these fields. Both spectra remain available in the JSON report.
    """
    available = report["spectra"]
    missing = [step for step in steps if str(step) not in available]
    if missing:
        raise SystemExit(
            f"steps {missing} not in report; available: {', '.join(sorted(available))}"
        )
    wavenumber = np.asarray(report["wavenumber"], dtype=float)

    with plt.rc_context(RC_PARAMS):
        figure, axes = plt.subplots(
            2,
            len(steps),
            figsize=(3.6 * len(steps) + 0.8, 6.4),
            layout="constrained",
            sharex=True,
            sharey="row",
        )
        for column, step in enumerate(steps):
            spectra = available[str(step)]
            _plot_spectra(axes[0][column], wavenumber, spectra, "energy", step)
            _plot_ratios(axes[1][column], wavenumber, spectra, "enstrophy", step)
        for row in axes:
            for column, axis in enumerate(row):
                if column:
                    axis.set_ylabel("")
        axes[0][0].set_ylabel("$E(k)$")
        axes[1][0].set_ylabel("$Z(k)\\,/\\,Z(k)_{\\mathrm{DNS}}$")

        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside lower center",
            ncols=3,
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
    args = parser.parse_args(argv)

    report = load_report(args.report)
    written = render(report, args.output_base, args.steps)
    print(f"wrote {written}")
    print(json.dumps(report["spectral_distance"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
