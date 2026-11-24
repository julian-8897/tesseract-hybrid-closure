"""Render the reported closure on one held-out turbulence trajectory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from urllib.request import urlopen

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from matplotlib.image import AxesImage

ROOT = Path(__file__).resolve().parents[2]
FONT_SOURCE = (
    "https://raw.githubusercontent.com/IBM/plex/"
    "ba3013d72a8419a22c758cc51c44e04ac7ea565b/packages/"
    "plex-sans/fonts/complete/ttf/"
)
SEED = 10000
HORIZON = 500
STEPS = np.linspace(1, HORIZON, 232).round().astype(int)
POSTER_INDEX = int(np.argmin(abs(STEPS - 150)))
FPS = 24
WIDTH, HEIGHT, DPI = 1440, 1560, 100
ACCENT = "#0a5c99"
INK = "#24282c"
MAX_BYTES = 4_000_000
OUTPUT_NAMES = (
    "closure_poster.png",
    "closure_still.png",
    "closure.mp4",
    "closure.webm",
)


def simulate(main: Path) -> tuple[np.ndarray, ...]:
    import jax
    import jax.numpy as jnp
    import torch

    from tesseract_hybrid_closure.closure import apply_closure
    from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
    from tesseract_hybrid_closure.data import generate_reference_trajectory
    from tesseract_hybrid_closure.final_eval import load_candidate_params
    from tesseract_hybrid_closure.losses import closure_rollout
    from tesseract_hybrid_closure.solver import CoarseVorticityStepper

    jax.config.update("jax_platforms", "cpu")
    torch.set_num_threads(4)

    manifest = json.loads((ROOT / "docs/results/final-metrics.json").read_text())
    records = manifest["source_evidence"]["files"]
    selected = next(
        r for r in records if r["role"] == "selected_aposteriori_checkpoint"
    )
    record = next(
        r
        for r in records
        if r["role"] == "aposteriori_candidate_checkpoint"
        and r["path"] == selected["path"]
        and r["candidate"] == selected["candidate"]
    )
    checkpoint, params = load_candidate_params(
        main / record["path"], expected_sha256=record["sha256"]
    )
    stepper = CoarseVorticityStepper(SolverConfig(**checkpoint["solver_config"]))
    reference = generate_reference_trajectory(
        SEED, HORIZON, split="validation", config=DNSConfig(**checkpoint["dns_config"])
    )
    hybrid = closure_rollout(stepper, params, reference.initial_coarse, HORIZON)
    dns = jnp.concatenate((reference.initial_coarse[None], reference.targets))
    dns_before, dns_after = dns[STEPS - 1], dns[STEPS]
    base = jax.vmap(stepper)(dns_before)
    dns_increment = dns_after - base
    cnn_increment = jnp.float32(stepper.config.dt) * jax.lax.map(
        lambda state: apply_closure(params, state), base
    )
    fields = tuple(
        np.asarray(field)[:, 0]
        for field in (dns_after, hybrid[STEPS - 1], dns_increment, cnn_increment)
    )
    if any(
        field.dtype != np.float32 or not np.isfinite(field).all() for field in fields
    ):
        raise ValueError("All four fields must be finite and float32")
    # The first hybrid step and first diagnostic have identical initial states.
    np.testing.assert_allclose(base[0] + cnn_increment[0], hybrid[0], rtol=0, atol=4e-6)
    print(f"Verified checkpoint {record['sha256']}; validation seed {SEED}", flush=True)
    return fields


def pooled_limit(left: np.ndarray, right: np.ndarray, increment: float) -> float:
    """Round the pooled 99th percentile of absolute values up to a tick value."""
    magnitude = np.abs(np.stack((left, right)))
    limit = np.ceil(np.percentile(magnitude, 99) / increment) * increment
    print(
        f"Limits ±{limit:g}; saturated pixels "
        f"{np.mean(magnitude[0] > limit):.3%} / {np.mean(magnitude[1] > limit):.3%}",
        flush=True,
    )
    return float(limit)


def plex_font(directory: Path, weight: str, pixels: int) -> FontProperties:
    from matplotlib.font_manager import FontProperties

    name = f"IBMPlexSans-{weight}.ttf"
    path = directory / name
    with urlopen(FONT_SOURCE + name, timeout=30) as response:
        path.write_bytes(response.read())
    return FontProperties(fname=path, size=pixels * 72 / DPI)


def make_plate(
    fields: tuple[np.ndarray, ...], directory: Path
) -> tuple[Figure, list[AxesImage]]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor="white")
    FigureCanvasAgg(figure)
    panel_font = plex_font(directory, "Medium", 49)
    key_font = plex_font(directory, "Regular", 43)

    limits = (
        pooled_limit(*fields[:2], 1.0),
        pooled_limit(*fields[2:], 0.005),
    )
    panel_names = (
        "Filtered DNS",
        "Hybrid rollout",
        "DNS-implied correction",
        "CNN correction",
    )
    images = []
    for index, (field, label) in enumerate(zip(fields, panel_names, strict=True)):
        row, column = divmod(index, 2)
        x, top = (64, 744)[column], (75, 850)[row]
        size = 632
        figure.text(
            (x + size / 2) / WIDTH,
            1 - (top - 35) / HEIGHT,
            label,
            ha="center",
            va="center",
            color=(INK, ACCENT)[column],
            fontproperties=panel_font,
        )
        axes = figure.add_axes(
            (x / WIDTH, 1 - (top + size) / HEIGHT, size / WIDTH, size / HEIGHT)
        )
        axes.set_axis_off()
        limit = limits[row]
        images.append(
            axes.imshow(
                field[POSTER_INDEX],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                origin="lower",
                interpolation="bilinear",
                interpolation_stage="data",
            )
        )

    for top, label, limit in zip(
        (730, 1505), ("Vorticity", "Vorticity per step"), limits, strict=True
    ):
        swatch = figure.add_axes(
            (746 / WIDTH, 1 - (top + 24) / HEIGHT, 278 / WIDTH, 24 / HEIGHT)
        )
        swatch.imshow(
            np.linspace(-limit, limit, 256)[None, :],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        swatch.set_axis_off()
        for x, text, align in (
            (584, label, "right"),
            (730, f"−{limit:g}", "right"),
            (1040, f"+{limit:g}", "left"),
        ):
            figure.text(
                x / WIDTH,
                1 - (top + 11) / HEIGHT,
                text,
                ha=align,
                va="center",
                color=INK,
                fontproperties=key_font,
            )
    return figure, images


def encode(
    figure: Figure, images: list[AxesImage], fields: tuple[np.ndarray, ...], out: Path
) -> None:
    """Feed both web encoders from one stream of RGB frames."""
    for image in images:
        image.set_animated(True)
    canvas = figure.canvas
    canvas.draw()
    background = canvas.copy_from_bbox(figure.bbox)

    def frame(index: int) -> np.ndarray:
        canvas.restore_region(background)
        for image, field in zip(images, fields, strict=True):
            image.set_data(field[index])
            image.axes.draw_artist(image)
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

    # Hold both endpoints briefly and fade through white at the loop reset.
    start, end = frame(0), frame(len(STEPS) - 1)
    fade_in = [255 - ((255 - start.astype(np.uint16)) * i // 6) for i in range(6)]
    fade_out = [
        255 - ((255 - end.astype(np.uint16)) * i // 6) for i in range(5, -1, -1)
    ]

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{WIDTH}x{HEIGHT}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
        "-filter_complex_threads",
        "2",
        "-filter_complex",
        "scale=out_color_matrix=bt709,format=yuv420p,split=2[mp4][webm]",
    ]
    for tag, codec_args in (
        (
            "mp4",
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "22",
                "-movflags",
                "+faststart",
            ],
        ),
        (
            "webm",
            [
                "-c:v",
                "libvpx-vp9",
                "-deadline",
                "good",
                "-cpu-used",
                "3",
                "-row-mt",
                "1",
                "-crf",
                "26",
                "-b:v",
                "2350k",
            ],
        ),
    ):
        command.extend(
            [
                "-map",
                f"[{tag}]",
                "-an",
                "-threads:v",
                "4",
                "-g",
                str(2 * FPS),
                "-maxrate",
                "2450k",
                "-bufsize",
                "1100k",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                *codec_args,
                str(out / f"closure.{tag}"),
            ]
        )

    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        with encoder.stdin as stream:
            for pixels in fade_in:
                stream.write(pixels.astype(np.uint8).tobytes())
            stream.write(start.tobytes() * 5)
            for index in range(len(STEPS)):
                stream.write(frame(index).tobytes())
            stream.write(end.tobytes() * 8)
            for pixels in fade_out:
                stream.write(pixels.astype(np.uint8).tobytes())
        status = encoder.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)
    for tag in ("mp4", "webm"):
        video = out / f"closure.{tag}"
        size = video.stat().st_size
        if size >= MAX_BYTES:
            raise ValueError(f"{video.name} exceeds the 4 MB budget: {size} bytes")
        print(f"{video.name}: {size:,} bytes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main", type=Path, default=ROOT.with_name("tesseract-hybrid-closure-agent")
    )
    args = parser.parse_args()

    out = ROOT / "portfolio_media"
    out.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=out, prefix=".closure-") as temp:
        staging = Path(temp)
        os.environ.setdefault("MPLCONFIGDIR", str(staging / "mpl-cache"))
        fields = simulate(args.main.resolve())
        plate, images = make_plate(fields, staging)
        for name, dpi in zip(OUTPUT_NAMES[:2], (DPI, 2 * DPI), strict=True):
            plate.savefig(staging / name, dpi=dpi)
        print(f"Poster/still step {STEPS[POSTER_INDEX]}; encoding video", flush=True)
        encode(plate, images, fields, staging)
        for name in OUTPUT_NAMES:
            (staging / name).replace(out / name)


if __name__ == "__main__":
    main()
