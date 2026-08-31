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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
HERO_DISPLAY_QUANTILE = 0.995
HERO_ZOOM_SIZE = 26
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


def _dns_temporal_zoom(
    dns: np.ndarray,
    *,
    size: int = HERO_ZOOM_SIZE,
) -> tuple[slice, slice]:
    """Select the crop with greatest mean DNS change, independent of predictions."""
    frames = np.stack(
        [assets.frame_2d(dns, step) for step in range(dns.shape[0])],
        axis=0,
    )
    change = np.mean((frames - frames[0]) ** 2, axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(change, (size, size))
    scores = np.sum(windows, axis=(-2, -1))
    top, left = np.unravel_index(np.argmax(scores), scores.shape)
    return slice(int(top), int(top + size)), slice(int(left), int(left + size))


def render_hero_animation(
    simulation: Mapping,
    figures_dir: Path,
) -> tuple[Path, Path, Path]:
    """Render a smooth web hero, GIF fallback and representative poster."""
    dns = np.asarray(simulation["dns_frames"])
    rollouts = simulation["rollouts"]
    aposteriori = np.asarray(rollouts["aposteriori-selected"])
    apriori = np.asarray(rollouts["apriori-matched"])
    no_closure = np.asarray(rollouts["no-closure"])
    field_data = [dns, aposteriori, no_closure]
    predictions = [aposteriori, apriori, no_closure]
    errors = [np.abs(prediction - dns) for prediction in predictions]

    # A few isolated extrema otherwise leave most of each diverging map nearly
    # white. One pooled, fixed robust scale keeps all columns comparable while
    # making the evolving structures legible. This affects display only.
    field_vmax = float(
        np.quantile(np.abs(np.concatenate(field_data, axis=0)), HERO_DISPLAY_QUANTILE)
    )
    error_vmax = float(
        np.quantile(np.concatenate(errors, axis=0), HERO_DISPLAY_QUANTILE)
    )
    zoom_y, zoom_x = _dns_temporal_zoom(dns)
    zoom_size = zoom_y.stop - zoom_y.start
    magnification = dns.shape[-1] / zoom_size

    figure = plt.figure(figsize=(11.4, 6.36), facecolor="#F7F8FA")
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.055],
        left=0.035,
        right=0.94,
        bottom=0.16,
        top=0.935,
        wspace=0.075,
        hspace=0.19,
    )
    top = [figure.add_subplot(grid[0, column]) for column in range(3)]
    bottom = [figure.add_subplot(grid[1, column]) for column in range(3)]
    field_labels = [
        "Filtered DNS",
        "Hybrid · JAX + PyTorch",
        "Solver only",
    ]
    field_accents = ["#475569", OKABE_ITO["blue"], "#64748B"]
    field_images = []
    overview_images = []
    for axis, data, label, accent in zip(
        top, field_data, field_labels, field_accents, strict=True
    ):
        frame = assets.frame_2d(data, 0)
        image = axis.imshow(
            frame[zoom_y, zoom_x],
            cmap="RdBu_r",
            vmin=-field_vmax,
            vmax=field_vmax,
            origin="lower",
            interpolation="bicubic",
        )
        field_images.append(image)
        axis.set_title(
            label,
            fontsize=12,
            fontweight="bold",
            color="#18212F",
            pad=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(0.8)
        axis.plot(
            [0, 1],
            [1.035, 1.035],
            transform=axis.transAxes,
            color=accent,
            linewidth=3.2,
            solid_capstyle="round",
            clip_on=False,
        )
        axis.text(
            0.035,
            0.96,
            f"DNS-selected temporal detail  ×{magnification:.1f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#17212B",
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "alpha": 0.78,
                "edgecolor": "none",
                "pad": 1.8,
            },
        )
        overview = axis.inset_axes([0.69, 0.035, 0.28, 0.28], zorder=5)
        overview_image = overview.imshow(
            frame,
            cmap="RdBu_r",
            vmin=-field_vmax,
            vmax=field_vmax,
            origin="lower",
            interpolation="bicubic",
        )
        overview_images.append(overview_image)
        overview.add_patch(
            Rectangle(
                (zoom_x.start - 0.5, zoom_y.start - 0.5),
                zoom_size,
                zoom_size,
                fill=False,
                edgecolor="#17212B",
                linewidth=1.2,
                linestyle=(0, (2, 2)),
            )
        )
        overview.set_xticks([])
        overview.set_yticks([])
        for spine in overview.spines.values():
            spine.set_color("white")
            spine.set_linewidth(2.3)
    field_colourbar = figure.colorbar(
        field_images[0],
        cax=figure.add_subplot(grid[0, 3]),
    )
    field_colourbar.set_label("vorticity  ω", fontsize=10)
    field_colourbar.ax.tick_params(labelsize=8)

    error_labels = [
        "Hybrid − DNS",
        "A-priori − DNS",
        "Solver − DNS",
    ]
    error_colours = [OKABE_ITO["blue"], OKABE_ITO["vermillion"], "#475569"]
    error_images = []
    score_texts = []
    for axis, data, label, colour in zip(
        bottom, errors, error_labels, error_colours, strict=True
    ):
        image = axis.imshow(
            assets.frame_2d(data, 0),
            cmap="magma",
            vmin=0,
            vmax=error_vmax,
            origin="lower",
            interpolation="bilinear",
        )
        error_images.append(image)
        axis.set_title(
            label,
            fontsize=11,
            fontweight="bold",
            color="#18212F",
            pad=7,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(0.8)
        score_texts.append(
            axis.text(
                0.5,
                -0.085,
                "rollout MSE  0.000",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9.5,
                color=colour,
                fontweight="bold",
            )
        )
    error_colourbar = figure.colorbar(
        error_images[0],
        cax=figure.add_subplot(grid[1, 3]),
    )
    error_colourbar.set_label("absolute error  |Δω|", fontsize=10)
    error_colourbar.ax.tick_params(labelsize=8)

    counter = figure.text(
        0.5,
        0.072,
        "step 000 / 500    ·    t = 0.000",
        ha="center",
        va="center",
        fontsize=11,
        color="#18212F",
        fontweight="bold",
    )
    progress_track = Line2D(
        [0.32, 0.68],
        [0.04, 0.04],
        transform=figure.transFigure,
        color="#D6DCE3",
        linewidth=5,
        solid_capstyle="round",
    )
    progress_line = Line2D(
        [0.32, 0.32],
        [0.04, 0.04],
        transform=figure.transFigure,
        color=OKABE_ITO["blue"],
        linewidth=5,
        solid_capstyle="round",
    )
    progress_dot = Line2D(
        [0.32],
        [0.04],
        transform=figure.transFigure,
        color=OKABE_ITO["blue"],
        marker="o",
        markersize=7,
        linestyle="none",
    )
    figure.add_artist(progress_track)
    figure.add_artist(progress_line)
    figure.add_artist(progress_dot)

    cumulative_mse = []
    for prediction in predictions:
        squared_error = (prediction[1:] - dns[1:]) ** 2
        per_step_mse = np.mean(
            squared_error,
            axis=tuple(range(1, squared_error.ndim)),
            dtype=np.float64,
        )
        cumulative_mse.append(
            np.concatenate(
                [
                    np.zeros(1),
                    np.cumsum(per_step_mse, dtype=np.float64)
                    / np.arange(1, SIMULATION_MAX_STEPS + 1),
                ]
            )
        )

    def update(step: int) -> list:
        for image, overview_image, data in zip(
            field_images, overview_images, field_data, strict=True
        ):
            frame = assets.frame_2d(data, step)
            image.set_data(frame[zoom_y, zoom_x])
            overview_image.set_data(frame)
        for image, data in zip(error_images, errors, strict=True):
            image.set_data(assets.frame_2d(data, step))
        for text, by_step in zip(score_texts, cumulative_mse, strict=True):
            text.set_text(f"rollout MSE  {by_step[step]:.3f}")
        counter.set_text(
            f"step {step:03d} / {SIMULATION_MAX_STEPS}    ·    t = {step * 0.002:.3f}"
        )
        progress_x = 0.32 + 0.36 * step / SIMULATION_MAX_STEPS
        progress_line.set_xdata([0.32, progress_x])
        progress_dot.set_xdata([progress_x])
        return [
            *field_images,
            *overview_images,
            *error_images,
            *score_texts,
            counter,
            progress_line,
            progress_dot,
        ]

    video_steps = (
        [0] * 11
        + list(range(0, SIMULATION_MAX_STEPS, ANIMATION_VIDEO_STRIDE))
        + [SIMULATION_MAX_STEPS] * 19
    )
    video_path = figures_dir / "hero_hybrid_rollout.mp4"
    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            "ffmpeg is required to render the smooth MP4 hero; install it or "
            "run with --no-simulation"
        )
    video = FuncAnimation(figure, update, frames=video_steps, blit=False)
    video.save(
        video_path,
        writer=FFMpegWriter(
            fps=ANIMATION_VIDEO_FPS,
            codec="libx264",
            bitrate=-1,
            metadata={"title": "Locked seed-20000 hybrid closure rollout"},
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
        dpi=100,
    )

    gif_steps = (
        [0] * 8
        + list(range(ANIMATION_GIF_STRIDE, SIMULATION_MAX_STEPS, ANIMATION_GIF_STRIDE))
        + [SIMULATION_MAX_STEPS] * 12
    )
    gif_path = figures_dir / "hero_hybrid_rollout.gif"
    gif = FuncAnimation(figure, update, frames=gif_steps, blit=False)
    # The GIF is a fallback for renderers without video; a smaller canvas
    # keeps it responsive in README embeds while the MP4 stays full-res.
    gif.save(gif_path, writer=PillowWriter(fps=ANIMATION_GIF_FPS), dpi=72)

    update(SIMULATION_MAX_STEPS)
    poster_path = figures_dir / "hero_hybrid_rollout_poster.png"
    figure.savefig(
        poster_path,
        dpi=100,
        facecolor=figure.get_facecolor(),
        metadata={},
    )
    plt.close(figure)
    return video_path, gif_path, poster_path


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
