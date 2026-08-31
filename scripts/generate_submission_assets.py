#!/usr/bin/env python
"""Generate the tracked submission result assets from the locked final protocol.

Reads only the sealed selection/test-evaluation reports and the persisted
training checkpoints (all inside ``runs/``, opened read-only) and writes only
its own generated outputs under ``docs/results/`` and ``docs/figures/``. The
script refuses any output path inside ``runs/``, ``artifacts/`` or ``.git``.
The curated metrics payload embeds the full per-seed error table plus a
post-evaluation integrity manifest (streamed SHA-256 and byte sizes) over the
preserved sources it consumed.

Determinism: fixed secrets-free generation, fixed matplotlib rcParams, no
timestamps in SVG/PNG metadata, and all simulated fields for the montage and
GIF animation are generated once per seed and reused.

Usage (from the repository root):

    uv run python scripts/generate_submission_assets.py
"""

from __future__ import annotations

import os

# XLA flags are read once when JAX first initialises, so this must run before
# any jax import below. Pinning Eigen to a single thread removes the
# nondeterministic multithreaded FFT reduction order, which otherwise perturbs
# the chaotic seed-20000 fields and breaks byte-determinism of the montage and
# animation outputs across runs.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_cpu_multi_thread_eigen=false"
).strip()

import argparse
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.font_manager import fontManager as _font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from tesseract_hybrid_closure import submission_assets as assets
from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
from tesseract_hybrid_closure.data import generate_reference_trajectory
from tesseract_hybrid_closure.final_eval import (
    FINAL_EVALUATION_HORIZONS,
    load_candidate_params,
    validate_selection_report_legacy_structure,
)
from tesseract_hybrid_closure.losses import (
    closure_rollout,
    no_closure_rollout,
    vorticity_mse,
)
from tesseract_hybrid_closure.solver import CoarseVorticityStepper

DEFAULT_COMMAND = "uv run python scripts/generate_submission_assets.py"

OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "graphite": "#000000",
    "yellow": "#F0E442",
    "reddish_purple": "#CC79A7",
}

METHOD_STYLE = {
    "aposteriori-selected": {
        "label": "A-posteriori (selected)",
        "colour": OKABE_ITO["blue"],
        "marker": "o",
        "linestyle": "-",
    },
    "apriori-matched": {
        "label": "A-priori (matched)",
        "colour": OKABE_ITO["vermillion"],
        "marker": "s",
        "linestyle": "--",
    },
    "no-closure": {
        "label": "No closure",
        "colour": OKABE_ITO["bluish_green"],
        "marker": "^",
        "linestyle": "-.",
    },
    "dynamic-smagorinsky": {
        "label": "Dynamic Smagorinsky",
        "colour": OKABE_ITO["orange"],
        "marker": "D",
        "linestyle": ":",
    },
    "static-smagorinsky": {
        "label": "Static Smagorinsky",
        "colour": OKABE_ITO["graphite"],
        "marker": "v",
        "linestyle": (0, (5, 2)),
    },
}

SIMULATION_SEED = 20_000
SIMULATION_MAX_STEPS = 500
MONTAGE_STEPS = (0, 30, 120, 500)
ANIMATION_GIF_STRIDE = 5
ANIMATION_GIF_FPS = 15
ANIMATION_VIDEO_STRIDE = 3
ANIMATION_VIDEO_FPS = 24
REL_TOL_FIELD_MSE = 5.0e-3
ABS_TOL_FIELD_MSE = 1.0e-7
PARAMETER_COUNT = 822977


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _report_paths(root: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    selection = Path(args.selection_report)
    evaluation = Path(args.evaluation_report)
    if not selection.is_absolute():
        selection = root / selection
    if not evaluation.is_absolute():
        evaluation = root / evaluation
    return selection, evaluation


def _resolve_output(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _guard_output_path(path: Path, root: Path, what: str) -> Path:
    """Allow writes only to this script's generated outputs under docs/."""
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{what} resolves outside the repository: {path}") from error
    if not relative.parts or relative.parts[0] != "docs":
        raise ValueError(f"refusing to write {what} outside docs/: {path}")
    return resolved


def _normalise_svg(path: Path) -> None:
    """Normalise the generated SVG into a stable, diff-friendly form.

    The SVG backend emits process-randomised hash ids (``id="p…"`` clip paths
    and ``id="image…"``) and leaves trailing spaces on long path lines.
    Rewriting both deterministically (stable sequential ids, base64 payloads
    on one line, LF line endings, no trailing whitespace, final newline
    preserved) makes the bytes reproducible across runs, well-formed for
    GitHub's renderer, and clean for ``git diff --check``.
    """
    text = path.read_text(encoding="utf-8")

    # Unwrap base64 payloads onto a single line. The previous wrapping broke
    # the payload across lines inside the quoted attribute: XML parsers
    # normalise those line breaks to spaces, corrupting the base64, and the
    # rewrap regex also consumed the attribute's closing quote, leaving the
    # document malformed. A single long attribute value is valid XML and
    # renders correctly.
    def rewrap_base64(match: re.Match) -> str:
        payload = "".join(match.group(1).split())
        return f'base64,{payload}"'

    text = re.sub(r"base64,\n([A-Za-z0-9+/=\n]+?)\"", rewrap_base64, text)

    # Stable sequential names for the hash-based ids (clip paths ``p…``,
    # spline paths ``m…``, raster ``image…``) and their references,
    # preserving document order of first appearance.
    renamed: dict[tuple[str, str], str] = {}

    def stable_name(kind: str, hexdigest: str) -> str:
        key = (kind, hexdigest)
        if key not in renamed:
            renamed[key] = f"{kind}{len(renamed)}"
        return renamed[key]

    def rename_with(factory) -> Callable[[re.Match], str]:
        def rename(match: re.Match) -> str:
            kind, hexdigest = match.group("kind"), match.group("hex")
            name = stable_name(kind, hexdigest)
            return factory(name)

        return rename

    text = re.sub(
        r'(?<![A-Za-z0-9_-])id="(?P<kind>image|m|p)(?P<hex>[0-9a-f]{8,})"',
        rename_with(lambda name: f'id="{name}"'),
        text,
    )
    text = re.sub(
        r"url\(#(?P<kind>image|m|p)(?P<hex>[0-9a-f]{8,})",
        rename_with(lambda name: f"url(#{name})"),
        text,
    )
    text = re.sub(
        r'xlink:href="#(?P<kind>image|m|p)(?P<hex>[0-9a-f]{8,})"',
        rename_with(lambda name: f'xlink:href="#{name}"'),
        text,
    )

    normalised = "\n".join(line.rstrip(" \t") for line in text.splitlines()) + "\n"
    path.write_text(normalised, encoding="utf-8")


def _save_figure(figure: plt.Figure, base: Path, *, dpi: int = 200) -> None:
    """Deterministic save: fixed dpi, no SVG timestamp metadata."""
    for suffix in ("svg", "png"):
        destination = base.with_suffix(f".{suffix}")
        figure.savefig(
            destination,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            metadata={"Date": None} if suffix == "svg" else {},
        )
        if suffix == "svg":
            _normalise_svg(destination)


def _write_caption(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def render_tesseract_results(
    figures_dir: Path,
    method_metrics: Mapping[str, Mapping[str, Mapping]],
    per_seed_errors: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Path:
    """Solver-gradient comparison and paired seed-500 errors."""
    horizons = np.asarray(FINAL_EVALUATION_HORIZONS)
    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.7),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )
    series = {
        "aposteriori-selected": (
            "A-posteriori · solver VJP",
            OKABE_ITO["blue"],
            "o",
            "-",
        ),
        "apriori-matched": (
            "A-priori · no solver gradients",
            OKABE_ITO["vermillion"],
            "s",
            "--",
        ),
        "no-closure": ("No closure", "#777777", "^", "-."),
    }
    for method, (label, colour, marker, linestyle) in series.items():
        means = np.asarray(
            [
                method_metrics[method][str(horizon)]["mean_vorticity_mse"]
                for horizon in horizons
            ]
        )
        stds = np.asarray(
            [
                method_metrics[method][str(horizon)]["std_vorticity_mse"]
                for horizon in horizons
            ]
        )
        left.plot(
            horizons,
            means,
            label=label,
            color=colour,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2 if method != "no-closure" else 1.4,
            markersize=6,
            alpha=1.0 if method != "no-closure" else 0.7,
        )
        left.fill_between(
            horizons,
            np.maximum(means - stds, 1.0e-6),
            means + stds,
            color=colour,
            alpha=0.11,
            linewidth=0,
        )
    left.set_yscale("log")
    left.set_xlabel("rollout horizon (coarse steps)")
    left.set_ylabel("vorticity MSE")
    left.set_xticks(horizons)
    left.legend(frameon=False, fontsize=8.8, loc="lower right")
    left.spines[["top", "right"]].set_visible(False)

    seed_keys = sorted(per_seed_errors, key=int)
    aposteriori_500 = np.asarray(
        [per_seed_errors[seed]["aposteriori-selected"]["500"] for seed in seed_keys]
    )
    apriori_500 = np.asarray(
        [per_seed_errors[seed]["apriori-matched"]["500"] for seed in seed_keys]
    )
    limits = (
        min(float(aposteriori_500.min()), float(apriori_500.min())) * 0.95,
        max(float(aposteriori_500.max()), float(apriori_500.max())) * 1.05,
    )
    right.plot(limits, limits, color="#777777", linestyle="--", linewidth=1.1)
    right.scatter(
        apriori_500,
        aposteriori_500,
        s=40,
        color=OKABE_ITO["blue"],
        edgecolor="white",
        linewidth=0.5,
    )
    right.set_xlim(limits)
    right.set_ylim(limits)
    right.set_aspect("equal", "box")
    right.set_xlabel("A-priori MSE · no solver gradients")
    right.set_ylabel("A-posteriori MSE · solver VJP")
    right.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    base = figures_dir / "fig_tesseract_results"
    _save_figure(figure, base)
    plt.close(figure)
    return base


def _simulate_seed20000(
    solver_config: SolverConfig,
    dns_config: DNSConfig,
    aposteriori_params: jnp.ndarray,
    apriori_params: jnp.ndarray,
) -> dict:
    """Generate the DNS reference and all method rollouts once, seed 20000."""
    stepper = CoarseVorticityStepper(solver_config)
    reference = generate_reference_trajectory(
        SIMULATION_SEED,
        SIMULATION_MAX_STEPS,
        split="test",
        config=dns_config,
    )
    rollouts = {
        "aposteriori-selected": assets.aligned_frames(
            reference.initial_coarse,
            closure_rollout(
                stepper,
                aposteriori_params,
                reference.initial_coarse,
                SIMULATION_MAX_STEPS,
            ),
        ),
        "apriori-matched": assets.aligned_frames(
            reference.initial_coarse,
            closure_rollout(
                stepper,
                apriori_params,
                reference.initial_coarse,
                SIMULATION_MAX_STEPS,
            ),
        ),
        "no-closure": assets.aligned_frames(
            reference.initial_coarse,
            no_closure_rollout(stepper, reference.initial_coarse, SIMULATION_MAX_STEPS),
        ),
    }
    dns_frames = assets.aligned_frames(reference.initial_coarse, reference.targets)
    return {
        "dns_frames": np.asarray(dns_frames, dtype=np.float32),
        "targets": np.asarray(reference.targets, dtype=np.float32),
        "rollouts": {
            name: np.asarray(trajectory, dtype=np.float32)
            for name, trajectory in rollouts.items()
        },
    }


def _verify_seed20000_mse(
    evaluation: Mapping,
    simulation: Mapping,
) -> dict[str, dict[str, float]]:
    """Verify regenerated trajectories reproduce the reported per-seed MSE."""
    reported = evaluation["per_seed_errors"][str(SIMULATION_SEED)]
    verified: dict[str, dict[str, float]] = {}
    for method in ("aposteriori-selected", "apriori-matched", "no-closure"):
        recomputed: dict[str, float] = {}
        for horizon in FINAL_EVALUATION_HORIZONS:
            expected = float(reported[method][str(horizon)])
            actual = float(
                vorticity_mse(
                    simulation["rollouts"][method][1 : horizon + 1],
                    simulation["targets"][:horizon],
                )
            )
            if not math.isclose(
                actual,
                expected,
                rel_tol=REL_TOL_FIELD_MSE,
                abs_tol=ABS_TOL_FIELD_MSE,
            ):
                raise ValueError(
                    f"{method} seed {SIMULATION_SEED} h={horizon}: recomputed "
                    f"MSE {actual:.8g} differs from the reported {expected:.8g}"
                )
            recomputed[str(horizon)] = actual
        verified[method] = recomputed
    return verified


# ---------------------------------------------------------------------------
# Instrument-panel hero (light-on-white). The design shows the three fields as
# a filmstrip, a signed "correction map" painted in the method colours (blue =
# hybrid closer to DNS, vermillion = solver closer), and a spec card with the
# headline rollout-MSE ratio and a live cumulative-MSE sparkline. Display colour
# scales are robust percentiles and affect display only, never a reported number.
# ---------------------------------------------------------------------------

_INSTALLED_FONTS = {face.name for face in _font_manager.ttflist}
_HERO_DISPLAY = FontProperties(
    family="DIN Condensed" if "DIN Condensed" in _INSTALLED_FONTS else "DejaVu Sans",
    weight="bold",
)


def _hero_mono(weight: str = "regular") -> FontProperties:
    family = (
        "JetBrains Mono" if "JetBrains Mono" in _INSTALLED_FONTS else "DejaVu Sans Mono"
    )
    return FontProperties(family=family, weight=weight)


_HERO_THEME = {
    "bg": "#F4F6F8",
    "tile": "#FBFCFD",
    "card": "#FFFFFF",
    "ink": "#141A22",
    "mut": "#5C6672",
    "faint": "#98A2AE",
    "rule": "#E2E7EC",
    "blue": OKABE_ITO["blue"],
    "diff_mid": "#FFFFFF",
}
_HERO_DIFF_ENDS = ("#E0662E", "#E9A87E", "#7FC4E8", OKABE_ITO["blue"])
HERO_DISPLAY_QUANTILE = 0.985


def _hero_diff_cmap(mid: str) -> LinearSegmentedColormap:
    verm, verm_lo, blue_lo, blue = _HERO_DIFF_ENDS
    return LinearSegmentedColormap.from_list(
        "hybrid_diff", [verm, verm_lo, mid, blue_lo, blue], N=256
    )


def _hero_context(simulation: Mapping, figW: float, figH: float) -> dict:
    """Fields, robust display scales and the fixed full-rollout headline ratio."""
    dns = np.asarray(simulation["dns_frames"])[:, 0]
    apo = np.asarray(simulation["rollouts"]["aposteriori-selected"])[:, 0]
    noc = np.asarray(simulation["rollouts"]["no-closure"])[:, 0]

    def cumulative(pred: np.ndarray) -> np.ndarray:
        squared = (pred[1:] - dns[1:]) ** 2
        per_step = squared.mean(axis=(1, 2), dtype=np.float64)
        return np.concatenate(
            [np.zeros(1), np.cumsum(per_step) / np.arange(1, SIMULATION_MAX_STEPS + 1)]
        )

    mse_apo = cumulative(apo)
    mse_noc = cumulative(noc)
    diff = np.abs(noc - dns) - np.abs(apo - dns)  # >0 => hybrid closer to DNS
    return {
        "dns": dns,
        "apo": apo,
        "noc": noc,
        "mse_apo": mse_apo,
        "mse_noc": mse_noc,
        "diff": diff,
        "field_vmax": float(
            np.quantile(np.abs(np.stack([dns, apo, noc])), HERO_DISPLAY_QUANTILE)
        ),
        "diff_vmax": float(np.quantile(np.abs(diff), HERO_DISPLAY_QUANTILE)),
        "headline": float(
            mse_noc[SIMULATION_MAX_STEPS] / mse_apo[SIMULATION_MAX_STEPS]
        ),
        "cmap": _hero_diff_cmap(_HERO_THEME["diff_mid"]),
        "T": _HERO_THEME,
        "figW": figW,
        "figH": figH,
    }


def _hero_tile_axes(fig, rect, T):
    axis = fig.add_axes(rect)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color(T["rule"])
        spine.set_linewidth(1.0)
    axis.set_facecolor(T["tile"])
    return axis


def _hero_multiplier(fig, number_text, x_gap, y, size, colour):
    """Place a small '×' just after a rendered number, measured from its extent."""
    box = number_text.get_window_extent(fig.canvas.get_renderer())
    x_right = fig.transFigure.inverted().transform((box.x1, 0))[0]
    fig.text(
        x_right + x_gap,
        y,
        "×",
        fontproperties=_HERO_DISPLAY,
        fontsize=size,
        color=colour,
        ha="left",
        va="center",
    )


def _hero_sparkline(fig, rect, ctx, step, T):
    axis = fig.add_axes(rect)
    axis.set_facecolor("none")
    full = np.arange(SIMULATION_MAX_STEPS + 1)
    past = full[: step + 1]
    axis.plot(full, ctx["mse_noc"], color=T["mut"], lw=1.2, alpha=0.32)
    axis.plot(full, ctx["mse_apo"], color=T["blue"], lw=1.2, alpha=0.32)
    axis.plot(past, ctx["mse_noc"][: step + 1], color=T["mut"], lw=1.9)
    axis.plot(past, ctx["mse_apo"][: step + 1], color=T["blue"], lw=2.4)
    axis.fill_between(past, ctx["mse_apo"][: step + 1], color=T["blue"], alpha=0.14)
    axis.plot([step], [ctx["mse_noc"][step]], "o", ms=4.5, color=T["mut"])
    axis.plot([step], [ctx["mse_apo"][step]], "o", ms=4.5, color=T["blue"])
    axis.set_xlim(0, SIMULATION_MAX_STEPS)
    axis.set_ylim(0, ctx["mse_noc"][SIMULATION_MAX_STEPS] * 1.12)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(T["rule"])
    axis.tick_params(colors=T["faint"], labelsize=7, length=2)
    axis.set_xticks([0, 250, 500])
    axis.set_yticks([])
    for label in axis.get_xticklabels():
        label.set_fontproperties(_hero_mono())


def _draw_hero_wide(fig, step: int, ctx: dict) -> None:
    """Landscape instrument panel (README hero)."""
    T = ctx["T"]
    A = ctx["figH"] / ctx["figW"]
    fig.set_facecolor(T["bg"])
    mono, mono_m, mono_b = _hero_mono(), _hero_mono("medium"), _hero_mono("bold")

    LX, LZONE_R = 0.035, 0.605
    PX0, PX1 = 0.648, 0.965

    fig.text(
        LX,
        0.952,
        "2D DECAYING TURBULENCE",
        fontproperties=mono_m,
        fontsize=10.5,
        color=T["mut"],
        ha="left",
        va="center",
    )
    fig.text(
        LX,
        0.903,
        "The hybrid recovers the fine-scale",
        fontproperties=_HERO_DISPLAY,
        fontsize=19,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        LX,
        0.866,
        "turbulence a coarse solver smears away.",
        fontproperties=_HERO_DISPLAY,
        fontsize=19,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        PX1,
        0.952,
        f"STEP {step:03d}/{SIMULATION_MAX_STEPS}   t={step * 0.002:.3f}",
        fontproperties=mono_m,
        fontsize=10,
        color=T["mut"],
        ha="right",
        va="center",
    )

    gap = 0.013
    fw = (LZONE_R - LX - 2 * gap) / 3
    fh = fw / A
    card_top = 0.815
    fy = card_top - fh
    labels = [
        ("FILTERED DNS", T["ink"]),
        ("HYBRID · JAX+PYTORCH", T["blue"]),
        ("SOLVER ONLY", T["ink"]),
    ]
    fields = [ctx["dns"], ctx["apo"], ctx["noc"]]
    field_image = None
    for i, (data, (label, colour)) in enumerate(zip(fields, labels, strict=True)):
        axis = _hero_tile_axes(fig, [LX + i * (fw + gap), fy, fw, fh], T)
        field_image = axis.imshow(
            assets.frame_2d(data, step),
            cmap="RdBu_r",
            vmin=-ctx["field_vmax"],
            vmax=ctx["field_vmax"],
            origin="lower",
            interpolation="bicubic",
        )
        axis.text(
            0.0,
            -0.055,
            label,
            transform=axis.transAxes,
            fontproperties=mono_m,
            fontsize=8.2,
            color=colour,
            ha="left",
            va="top",
        )

    cax = fig.add_axes([LZONE_R + 0.006, fy, 0.008, fh])
    bar = fig.colorbar(field_image, cax=cax)
    bar.outline.set_edgecolor(T["rule"])
    bar.outline.set_linewidth(0.8)
    bar.ax.tick_params(colors=T["mut"], labelsize=6.5, length=2)
    for label in bar.ax.get_yticklabels():
        label.set_fontproperties(mono)
    fig.text(
        LZONE_R + 0.01,
        fy + fh + 0.016,
        "ω",
        fontproperties=mono_m,
        fontsize=9.5,
        color=T["mut"],
        ha="center",
        va="center",
    )

    ch = 0.325
    cw = ch * A
    cy = 0.07
    axd = _hero_tile_axes(fig, [LX, cy, cw, ch], T)
    axd.imshow(
        ctx["diff"][step],
        cmap=ctx["cmap"],
        vmin=-ctx["diff_vmax"],
        vmax=ctx["diff_vmax"],
        origin="lower",
        interpolation="bicubic",
    )
    tx = LX + cw + 0.028
    fig.text(
        tx,
        cy + ch - 0.01,
        "CORRECTION",
        fontproperties=_HERO_DISPLAY,
        fontsize=25,
        color=T["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        tx,
        cy + ch - 0.075,
        "MAP",
        fontproperties=_HERO_DISPLAY,
        fontsize=25,
        color=T["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        tx,
        cy + ch - 0.135,
        "per-cell vorticity error the",
        fontproperties=mono,
        fontsize=8.2,
        color=T["mut"],
        ha="left",
        va="top",
    )
    fig.text(
        tx,
        cy + ch - 0.163,
        "hybrid removes vs the solver",
        fontproperties=mono,
        fontsize=8.2,
        color=T["mut"],
        ha="left",
        va="top",
    )
    for j, (swatch, text) in enumerate(
        [
            (OKABE_ITO["blue"], "hybrid closer to DNS"),
            ("#E0662E", "solver closer to DNS"),
        ]
    ):
        yy = cy + 0.055 - j * 0.038
        fig.add_artist(
            Rectangle(
                (tx, yy),
                0.016,
                0.024,
                transform=fig.transFigure,
                facecolor=swatch,
                edgecolor="none",
            )
        )
        fig.text(
            tx + 0.024,
            yy + 0.012,
            text,
            fontproperties=mono,
            fontsize=8,
            color=T["mut"],
            ha="left",
            va="center",
        )

    card = FancyBboxPatch(
        (PX0, cy),
        PX1 - PX0,
        card_top - cy,
        transform=fig.transFigure,
        boxstyle="round,pad=0,rounding_size=0.02",
        mutation_aspect=ctx["figW"] / ctx["figH"],
        facecolor=T["card"],
        edgecolor=T["rule"],
        linewidth=1.0,
        zorder=0,
    )
    fig.add_artist(card)
    ix0, ix1 = PX0 + 0.024, PX1 - 0.024
    fig.text(
        ix0,
        0.768,
        "RESULT",
        fontproperties=mono_m,
        fontsize=9.5,
        color=T["blue"],
        ha="left",
        va="center",
    )
    # Live cumulative-MSE ratio so the headline advances with the rollout and
    # equals the two metric rows below it (solver / hybrid). It converges to the
    # fixed full-rollout figure ctx["headline"] at step 500, which the poster
    # renders; the step-0 fallback avoids a 0/0 before any error accrues.
    apo_mse = ctx["mse_apo"][step]
    live_ratio = ctx["mse_noc"][step] / apo_mse if apo_mse > 0 else ctx["headline"]
    number = fig.text(
        ix0 - 0.004,
        0.652,
        f"{live_ratio:.2f}",
        fontproperties=_HERO_DISPLAY,
        fontsize=58,
        color=T["blue"],
        ha="left",
        va="center",
    )
    _hero_multiplier(fig, number, 0.012, 0.670, 22, T["blue"])
    fig.text(
        ix0,
        0.565,
        "LOWER ROLLOUT ERROR",
        fontproperties=mono_m,
        fontsize=9.5,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        ix0,
        0.537,
        "THAN THE SOLVER ALONE",
        fontproperties=mono_m,
        fontsize=9.5,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.add_artist(
        Line2D(
            [ix0, ix1],
            [0.498, 0.498],
            transform=fig.transFigure,
            color=T["rule"],
            lw=1.0,
        )
    )

    def metric_row(y, tag, value, colour):
        fig.text(
            ix0,
            y,
            tag,
            fontproperties=mono,
            fontsize=9,
            color=T["mut"],
            ha="left",
            va="center",
        )
        fig.text(
            ix1,
            y,
            value,
            fontproperties=mono_b,
            fontsize=15,
            color=colour,
            ha="right",
            va="center",
        )

    metric_row(0.455, "HYBRID  rollout MSE", f"{ctx['mse_apo'][step]:.3f}", T["blue"])
    metric_row(0.410, "SOLVER  rollout MSE", f"{ctx['mse_noc'][step]:.3f}", T["mut"])
    fig.text(
        ix0,
        0.340,
        "CUMULATIVE MSE OVER THE ROLLOUT",
        fontproperties=mono,
        fontsize=7.8,
        color=T["mut"],
        ha="left",
        va="center",
    )
    _hero_sparkline(fig, [ix0, 0.098, ix1 - ix0, 0.205], ctx, step, T)


def _draw_hero_portrait(fig, step: int, ctx: dict) -> None:
    """4:5 portrait instrument panel (LinkedIn)."""
    T = ctx["T"]
    A = ctx["figH"] / ctx["figW"]
    fig.set_facecolor(T["bg"])
    mono, mono_m, mono_b = _hero_mono(), _hero_mono("medium"), _hero_mono("bold")

    LX, RX = 0.05, 0.95
    fig.text(
        LX,
        0.966,
        "2D DECAYING TURBULENCE",
        fontproperties=mono_m,
        fontsize=11,
        color=T["mut"],
        ha="left",
        va="center",
    )
    fig.text(
        RX,
        0.966,
        f"STEP {step:03d}/{SIMULATION_MAX_STEPS}",
        fontproperties=mono_m,
        fontsize=10.5,
        color=T["mut"],
        ha="right",
        va="center",
    )
    fig.text(
        LX,
        0.930,
        "The hybrid recovers the fine-scale turbulence",
        fontproperties=_HERO_DISPLAY,
        fontsize=23,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        LX,
        0.902,
        "a coarse solver smears away.",
        fontproperties=_HERO_DISPLAY,
        fontsize=23,
        color=T["ink"],
        ha="left",
        va="center",
    )

    # field filmstrip
    gap = 0.02
    fw = (RX - LX - 2 * gap) / 3
    fh = fw / A
    fy = 0.855 - fh
    labels = [
        ("FILTERED DNS", T["ink"]),
        ("HYBRID · JAX+PYTORCH", T["blue"]),
        ("SOLVER ONLY", T["ink"]),
    ]
    fields = [ctx["dns"], ctx["apo"], ctx["noc"]]
    field_image = None
    for i, (data, (label, colour)) in enumerate(zip(fields, labels, strict=True)):
        axis = _hero_tile_axes(fig, [LX + i * (fw + gap), fy, fw, fh], T)
        field_image = axis.imshow(
            assets.frame_2d(data, step),
            cmap="RdBu_r",
            vmin=-ctx["field_vmax"],
            vmax=ctx["field_vmax"],
            origin="lower",
            interpolation="bicubic",
        )
        axis.text(
            0.0,
            -0.05,
            label,
            transform=axis.transAxes,
            fontproperties=mono_m,
            fontsize=9.2,
            color=colour,
            ha="left",
            va="top",
        )

    # horizontal vorticity colourbar under the strip
    cbar_y = fy - 0.05
    cax = fig.add_axes([0.35, cbar_y, 0.30, 0.011])
    bar = fig.colorbar(field_image, cax=cax, orientation="horizontal")
    bar.outline.set_edgecolor(T["rule"])
    bar.outline.set_linewidth(0.8)
    bar.ax.tick_params(colors=T["mut"], labelsize=7, length=2)
    for label in bar.ax.get_xticklabels():
        label.set_fontproperties(mono)
    fig.text(
        0.33,
        cbar_y + 0.005,
        "ω",
        fontproperties=mono_m,
        fontsize=10,
        color=T["mut"],
        ha="right",
        va="center",
    )

    # correction map (bottom-left) with title/caption/legend stacked above it
    ch = 0.33
    cw = ch * A
    cy = 0.075
    axd = _hero_tile_axes(fig, [LX, cy, cw, ch], T)
    axd.imshow(
        ctx["diff"][step],
        cmap=ctx["cmap"],
        vmin=-ctx["diff_vmax"],
        vmax=ctx["diff_vmax"],
        origin="lower",
        interpolation="bicubic",
    )
    top_left = cy + ch
    fig.text(
        LX,
        top_left + 0.135,
        "CORRECTION MAP",
        fontproperties=_HERO_DISPLAY,
        fontsize=22,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        LX,
        top_left + 0.098,
        "per-cell vorticity error the hybrid removes",
        fontproperties=mono,
        fontsize=8.6,
        color=T["mut"],
        ha="left",
        va="center",
    )
    for j, (swatch, text) in enumerate(
        [
            (OKABE_ITO["blue"], "hybrid closer to DNS"),
            ("#E0662E", "solver closer to DNS"),
        ]
    ):
        yy = top_left + 0.06 - j * 0.03
        fig.add_artist(
            Rectangle(
                (LX, yy - 0.009),
                0.018,
                0.018,
                transform=fig.transFigure,
                facecolor=swatch,
                edgecolor="none",
            )
        )
        fig.text(
            LX + 0.028,
            yy,
            text,
            fontproperties=mono,
            fontsize=8.4,
            color=T["mut"],
            ha="left",
            va="center",
        )

    # result card (bottom-right)
    card_left, card_right = LX + cw + 0.03, RX + 0.02
    card_bottom, card_top = cy - 0.02, 0.55
    fig.add_artist(
        FancyBboxPatch(
            (card_left, card_bottom),
            card_right - card_left,
            card_top - card_bottom,
            transform=fig.transFigure,
            boxstyle="round,pad=0,rounding_size=0.02",
            mutation_aspect=ctx["figW"] / ctx["figH"],
            facecolor=T["card"],
            edgecolor=T["rule"],
            linewidth=1.0,
            zorder=0,
        )
    )
    ix0, ix1 = card_left + 0.026, card_right - 0.026
    fig.text(
        ix0,
        0.515,
        "RESULT",
        fontproperties=mono_m,
        fontsize=10,
        color=T["blue"],
        ha="left",
        va="center",
    )
    number = fig.text(
        ix0 - 0.004,
        0.435,
        f"{ctx['headline']:.2f}",
        fontproperties=_HERO_DISPLAY,
        fontsize=62,
        color=T["blue"],
        ha="left",
        va="center",
    )
    _hero_multiplier(fig, number, 0.012, 0.457, 25, T["blue"])
    fig.text(
        ix0,
        0.352,
        "LOWER ROLLOUT ERROR THAN",
        fontproperties=mono_m,
        fontsize=10,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        ix0,
        0.324,
        "THE SOLVER ALONE",
        fontproperties=mono_m,
        fontsize=10,
        color=T["ink"],
        ha="left",
        va="center",
    )
    fig.add_artist(
        Line2D(
            [ix0, ix1],
            [0.285, 0.285],
            transform=fig.transFigure,
            color=T["rule"],
            lw=1.0,
        )
    )

    def metric_row(y, tag, value, colour):
        fig.text(
            ix0,
            y,
            tag,
            fontproperties=mono,
            fontsize=9.5,
            color=T["mut"],
            ha="left",
            va="center",
        )
        fig.text(
            ix1,
            y,
            value,
            fontproperties=mono_b,
            fontsize=15.5,
            color=colour,
            ha="right",
            va="center",
        )

    metric_row(0.246, "HYBRID  rollout MSE", f"{ctx['mse_apo'][step]:.3f}", T["blue"])
    metric_row(0.204, "SOLVER  rollout MSE", f"{ctx['mse_noc'][step]:.3f}", T["mut"])
    fig.text(
        ix0,
        0.152,
        "CUMULATIVE MSE OVER THE ROLLOUT",
        fontproperties=mono,
        fontsize=7.8,
        color=T["mut"],
        ha="left",
        va="center",
    )
    _hero_sparkline(fig, [ix0, 0.078, ix1 - ix0, 0.062], ctx, step, T)


def _render_hero(
    simulation: Mapping,
    figures_dir: Path,
    *,
    figsize: tuple[float, float],
    draw,
    stem: str,
    video_dpi: int,
    gif_dpi: int,
    gif_stride: int,
    title: str,
) -> tuple[Path, Path, Path]:
    """Render one instrument-panel layout to smooth MP4, GIF and poster."""
    ctx = _hero_context(simulation, figsize[0], figsize[1])
    figure = plt.figure(figsize=figsize, facecolor=_HERO_THEME["bg"])

    def update(step: int) -> list:
        figure.clf()
        draw(figure, step, ctx)
        return []

    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            "ffmpeg is required to render the smooth MP4 hero; install it or "
            "run with --no-simulation"
        )
    video_steps = (
        [0] * 8
        + list(range(0, SIMULATION_MAX_STEPS, ANIMATION_VIDEO_STRIDE))
        + [SIMULATION_MAX_STEPS] * 18
    )
    video_path = figures_dir / f"{stem}.mp4"
    FuncAnimation(figure, update, frames=video_steps, blit=False).save(
        video_path,
        writer=FFMpegWriter(
            fps=ANIMATION_VIDEO_FPS,
            codec="libx264",
            bitrate=-1,
            metadata={"title": title},
            extra_args=[
                "-preset",
                "slow",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-metadata",
                "creation_time=1970-01-01T00:00:00Z",
            ],
        ),
        dpi=video_dpi,
    )

    gif_steps = (
        [0] * 6
        + list(range(0, SIMULATION_MAX_STEPS, gif_stride))
        + [SIMULATION_MAX_STEPS] * 10
    )
    gif_path = figures_dir / f"{stem}.gif"
    FuncAnimation(figure, update, frames=gif_steps, blit=False).save(
        gif_path, writer=PillowWriter(fps=ANIMATION_GIF_FPS), dpi=gif_dpi
    )

    update(SIMULATION_MAX_STEPS)
    poster_path = figures_dir / f"{stem}_poster.png"
    figure.savefig(
        poster_path, dpi=video_dpi, facecolor=figure.get_facecolor(), metadata={}
    )
    plt.close(figure)
    return video_path, gif_path, poster_path


def render_hero_animation(
    simulation: Mapping,
    figures_dir: Path,
) -> tuple[Path, Path, Path]:
    """Render the landscape README hero (smooth MP4, GIF fallback, poster)."""
    return _render_hero(
        simulation,
        figures_dir,
        figsize=(12.8, 7.0),
        draw=_draw_hero_wide,
        stem="hero_hybrid_rollout",
        video_dpi=110,
        gif_dpi=92,
        gif_stride=9,
        title="Locked seed-20000 hybrid closure rollout",
    )


def render_hero_linkedin(
    simulation: Mapping,
    figures_dir: Path,
) -> tuple[Path, Path, Path]:
    """Render the 4:5 portrait LinkedIn hero (smooth MP4, GIF, poster)."""
    return _render_hero(
        simulation,
        figures_dir,
        figsize=(8.64, 10.8),
        draw=_draw_hero_portrait,
        stem="hero_linkedin",
        video_dpi=125,
        gif_dpi=66,
        gif_stride=6,
        title="Locked seed-20000 hybrid closure rollout (LinkedIn)",
    )


def render_local_error_reduction(simulation: Mapping, figures_dir: Path) -> Path:
    """Step-500 absolute errors and their signed local difference."""
    step = SIMULATION_MAX_STEPS
    dns = np.asarray(simulation["dns_frames"])
    aposteriori = np.asarray(simulation["rollouts"]["aposteriori-selected"])
    no_closure = np.asarray(simulation["rollouts"]["no-closure"])
    hybrid_error = assets.frame_2d(np.abs(aposteriori - dns), step)
    solver_error = assets.frame_2d(np.abs(no_closure - dns), step)
    reduction = solver_error - hybrid_error
    error_vmax = float(
        np.quantile(np.concatenate([hybrid_error.ravel(), solver_error.ravel()]), 0.995)
    )
    reduction_vmax = float(np.quantile(np.abs(reduction), 0.995))

    figure, axes = plt.subplots(1, 3, figsize=(9.7, 3.25), layout="constrained")
    absolute_image = axes[0].imshow(
        hybrid_error,
        cmap="magma",
        vmin=0,
        vmax=error_vmax,
        origin="lower",
        interpolation="nearest",
    )
    axes[1].imshow(
        solver_error,
        cmap="magma",
        vmin=0,
        vmax=error_vmax,
        origin="lower",
        interpolation="nearest",
    )
    reduction_image = axes[2].imshow(
        reduction,
        cmap="BrBG",
        vmin=-reduction_vmax,
        vmax=reduction_vmax,
        origin="lower",
        interpolation="nearest",
    )
    for label, axis in zip(("(a)", "(b)", "(c)"), axes, strict=True):
        axis.text(
            0.03,
            0.96,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(
        absolute_image,
        ax=axes[:2],
        shrink=0.88,
        pad=0.02,
        label="absolute error |Δω|",
    )
    figure.colorbar(
        reduction_image,
        ax=axes[2],
        shrink=0.88,
        pad=0.02,
        label="local error reduction",
    )
    base = figures_dir / "fig_local_error_reduction"
    _save_figure(figure, base, dpi=180)
    plt.close(figure)
    return base


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the tracked submission result assets from the locked "
            "final protocol (deterministic, read-only over runs/)."
        )
    )
    parser.add_argument(
        "--selection-report",
        default="runs/final-submission/selection.json",
        help="selection report (relative to the repository root by default)",
    )
    parser.add_argument(
        "--evaluation-report",
        default="runs/final-submission/test-evaluation.json",
        help="test-evaluation report (relative to the repository root by default)",
    )
    parser.add_argument(
        "--figures-dir",
        default="docs/figures",
        help="generated figures directory (must live inside docs/)",
    )
    parser.add_argument(
        "--metrics-output",
        default="docs/results/final-metrics.json",
        help="generated metrics JSON (must live inside docs/)",
    )
    parser.add_argument(
        "--no-simulation",
        action="store_true",
        help="skip the montage and animation (fields for seed 20000)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = _repo_root()
    selection_path, evaluation_path = _report_paths(root, args)
    figures_dir = _guard_output_path(
        _resolve_output(root, args.figures_dir), root, "figures directory"
    )
    metrics_output = _guard_output_path(
        _resolve_output(root, args.metrics_output), root, "metrics output"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] validating selection report {selection_path} ...")
    selection = _load_json(selection_path, "selection report")
    validate_selection_report_legacy_structure(selection)
    print(f"      selected {selection['selected']['name']}")

    print(f"[2/6] validating test-evaluation report {evaluation_path} ...")
    evaluation = _load_json(evaluation_path, "test-evaluation report")
    assets.validate_evaluation_report(evaluation, selection)
    print(
        f"      methods {sorted(evaluation['methods'])} internally consistent "
        f"(32 seeds, 5 horizons)"
    )

    print("[3/6] deriving curated metrics + source evidence ...")
    generation = {
        "command": DEFAULT_COMMAND,
        "script": assets.relative_repo_path(Path(__file__), root),
        "assets": [],
    }
    source_evidence = assets.build_source_evidence(
        root=root,
        selection_report_path=assets.relative_repo_path(selection_path, root),
        evaluation_report_path=assets.relative_repo_path(evaluation_path, root),
        selection=selection,
        evaluation=evaluation,
    )
    for record in source_evidence["files"]:
        label = record["role"]
        if "candidate" in record:
            label += f" ({record['candidate']})"
        digest = record.get("sha256")
        if digest is None:
            digest = f"duplicate of {record['ref']}"
        print(
            f"      {label}: {record['path']} "
            f"({record.get('bytes', '-')} bytes, sha256 {digest[:16]})"
        )
    metrics = assets.build_final_metrics(
        selection,
        evaluation,
        generation=generation,
        source_evidence=source_evidence,
    )
    assets.write_owned_json(metrics_output, metrics)
    print(f"      wrote {metrics_output}")

    method_metrics = assets.derive_method_metrics(evaluation)
    print("[4/6] rendering Tesseract-focused result figure ...")
    tesseract_results_base = render_tesseract_results(
        figures_dir,
        method_metrics,
        evaluation["per_seed_errors"],
    )
    _write_caption(
        figures_dir / "fig_tesseract_results_caption.txt",
        (
            "Left: rollout vorticity-MSE on the locked 32-seed test split. "
            "Markers show the mean and bands show +/- 1 population standard "
            "deviation. The matched CNNs share their architecture, seed-zero "
            "initialisation, Adam learning rate, 700-update budget, training "
            "seed budget and calibrated flow regime; only the a-posteriori "
            "objective differentiates through the solver rollout. Right: "
            "paired 500-step errors for the same 32 test seeds. Every point "
            "falls below the equal-error diagonal, so the a-posteriori model "
            "wins on all 32 seeds. Exact values are in "
            "docs/results/final-metrics.json."
        ),
    )
    print(f"      wrote {tesseract_results_base}.svg/.png + caption")

    hero_paths: tuple[Path, ...] = ()
    local_error_base: Path | None = None
    if not args.no_simulation:
        print("[5/6] simulating seed 20000 (DNS + 3 rollouts, 500 steps) ...")
        solver_config = SolverConfig(**evaluation["solver_config"])
        dns_config = DNSConfig(**evaluation["dns_config"])
        _, aposteriori_params = load_candidate_params(
            evaluation["selected_aposteriori_checkpoint"]
        )
        _, apriori_params = load_candidate_params(evaluation["apriori_checkpoint"])
        simulation = _simulate_seed20000(
            solver_config,
            dns_config,
            aposteriori_params,
            apriori_params,
        )
        verified = _verify_seed20000_mse(evaluation, simulation)
        print("      regenerated fields reproduce the reported per-seed MSE:")
        for method, by_horizon in verified.items():
            print(
                f"        {method}: "
                + ", ".join(
                    f"h{h}={value:.6g}" for h, value in sorted(by_horizon.items())
                )
            )
        hero_paths = render_hero_animation(simulation, figures_dir)
        hero_paths = hero_paths + render_hero_linkedin(simulation, figures_dir)
        print("      wrote " + ", ".join(str(path) for path in hero_paths))
        local_error_base = render_local_error_reduction(simulation, figures_dir)
        _write_caption(
            figures_dir / "fig_local_error_reduction_caption.txt",
            (
                "Spatial error comparison at step 500 for locked test seed "
                "20000, fixed before visualisation. Panels show (a) absolute "
                "error of the selected a-posteriori hybrid, (b) absolute "
                "error of the no-closure solver on the same linear scale, and "
                "(c) their signed difference, |omega_solver - omega_DNS| - "
                "|omega_hybrid - omega_DNS|. Positive green regions are local "
                "error reductions and negative brown regions are local "
                "regressions. Colour limits use the 99.5th percentile only "
                "for display; no values enter the reported metric. The "
                "locked metric remains rollout vorticity-MSE over all cells "
                "and times."
            ),
        )
        print(f"      wrote {local_error_base}.svg/.png + caption")

    print("[6/6] recording generated assets ...")
    for path in (
        metrics_output,
        tesseract_results_base.with_suffix(".svg"),
        tesseract_results_base.with_suffix(".png"),
        *hero_paths,
        local_error_base.with_suffix(".svg") if local_error_base else None,
        local_error_base.with_suffix(".png") if local_error_base else None,
    ):
        if path is not None:
            generation["assets"].append(assets.relative_repo_path(path, root))
    metrics["generation"]["assets"] = sorted(set(generation["assets"]))
    assets.write_owned_json(metrics_output, metrics)

    reductions = assets.derive_relative_reductions(method_metrics)
    paired = assets.derive_paired_wins(evaluation)
    print("[7/7] headlined verified numbers")
    print(
        f"      a-posteriori test MSE: "
        f"h30={method_metrics['aposteriori-selected']['30']['mean_vorticity_mse']:.6f}, "
        f"h500={method_metrics['aposteriori-selected']['500']['mean_vorticity_mse']:.6f}"
    )
    print(
        f"      reduction vs no closure: "
        f"h30={reductions['30']['vs_no_closure']:.3%}, "
        f"h500={reductions['500']['vs_no_closure']:.3%}"
    )
    print(
        f"      reduction vs best Smagorinsky: "
        f"h30={reductions['30']['vs_best_smagorinsky']:.3%}, "
        f"h500={reductions['500']['vs_best_smagorinsky']:.3%}"
    )
    print(
        "      paired wins vs a-priori (of 32 seeds): "
        + ", ".join(
            f"h{h}={paired[h]['aposteriori_wins']}"
            for h in (str(v) for v in FINAL_EVALUATION_HORIZONS)
        )
    )
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
