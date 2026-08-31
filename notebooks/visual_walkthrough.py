import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Differentiable hybrid closure")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
    from tesseract_hybrid_closure.data import generate_reference_trajectory
    from tesseract_hybrid_closure.final_eval import (
        load_candidate_params_with_digest,
    )
    from tesseract_hybrid_closure.losses import (
        closure_rollout,
        no_closure_rollout,
    )
    from tesseract_hybrid_closure.solver import CoarseVorticityStepper
    from tesseract_hybrid_closure.spectra import (
        coarse_grid_max_wavenumber,
        radial_spectra,
        spectral_distance,
    )

    return (
        CoarseVorticityStepper,
        DNSConfig,
        Path,
        SolverConfig,
        closure_rollout,
        coarse_grid_max_wavenumber,
        generate_reference_trajectory,
        jax,
        json,
        load_candidate_params_with_digest,
        mo,
        no_closure_rollout,
        np,
        plt,
        radial_spectra,
        spectral_distance,
    )


@app.cell(hide_code=True)
def _(Path):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_CHECKPOINT = (
        REPO_ROOT
        / "runs"
        / "w2-calibrated-a20-dt002-100x3"
        / "stage-unroll-30-updates-700.pkl"
    )
    EXPECTED_SHA256 = "8ed5b36fac902e61fcd3a1749f727f6928c93f3b7303e5781f5f66ce86c2e9b7"
    RELEASE_URL = (
        "https://github.com/julian-8897/tesseract-hybrid-closure/releases/"
        "download/submission-2026-08-31/stage-unroll-30-updates-700.pkl"
    )
    return DEFAULT_CHECKPOINT, EXPECTED_SHA256, RELEASE_URL, REPO_ROOT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Differentiable hybrid closure for 2D turbulence

    **A visual walkthrough of a PyTorch closure trained through a JAX/Exponax
    spectral solver.** The two frameworks meet on one reverse-mode AD tape via
    explicit VJP rules.

    This notebook uses one deterministic **validation** trajectory. It never
    accesses the sealed test split, retrains the model, or changes the locked
    numerical regime.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.hstack(
        [
            mo.stat(value="256² → 64²", label="DNS → coarse grid"),
            mo.stat(value="10 × 64 × 5²", label="PyTorch CNN"),
            mo.stat(value="822,977", label="trainable parameters"),
            mo.stat(value="1 → 5 → 30", label="unroll curriculum"),
            mo.stat(value="Δt = 0.002", label="locked timestep"),
        ],
        widths="equal",
        gap=1.0,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One forward model, one reverse path

    Each coarse step first advances vorticity with the ETDRK2 spectral solver,
    then applies an explicit learned tendency correction:

    ```text
    ωₙ ── JAX / Exponax coarse step ──► ω* ── + Δt · PyTorch CNN(ω*) ──► ωₙ₊₁
                  ▲                                                   │
                  └──────── solver VJP ◄── closure VJP ◄─────────────┘
    ```

    Training minimises rollout vorticity-MSE. Gradients cross the framework
    boundary through a `jax.custom_vjp` backed by PyTorch autograd callbacks.
    """)
    return


@app.cell(hide_code=True)
def _(DEFAULT_CHECKPOINT, mo):
    checkpoint_path = mo.ui.text(
        value=str(DEFAULT_CHECKPOINT),
        label="Submitted checkpoint",
        full_width=True,
    )
    validation_seed = mo.ui.dropdown(
        options={f"Validation seed {seed}": seed for seed in range(10000, 10004)},
        value="Validation seed 10000",
        label="Trajectory",
    )
    run_walkthrough = mo.ui.run_button(label="Generate 30-step walkthrough")
    mo.vstack(
        [
            mo.md("## 2. Generate a held-out trajectory"),
            mo.callout(
                "The first run compiles JAX and evolves a 256² DNS reference. "
                "It can take about a minute on CPU; later visual controls are immediate.",
                kind="info",
            ),
            checkpoint_path,
            mo.hstack([validation_seed, run_walkthrough], justify="start", gap=1.0),
        ],
        gap=0.75,
    )
    return checkpoint_path, run_walkthrough, validation_seed


@app.cell(hide_code=True)
def _(Path, RELEASE_URL, checkpoint_path, mo, run_walkthrough):
    mo.stop(
        not run_walkthrough.value,
        mo.callout(
            "Choose a validation seed, then generate the walkthrough.", kind="neutral"
        ),
    )
    _path = checkpoint_path.value.strip()
    mo.stop(
        not _path,
        mo.callout("Enter a checkpoint path before running.", kind="warn"),
    )
    _checkpoint = Path(_path).expanduser().resolve()
    mo.stop(
        not _checkpoint.is_file(),
        mo.callout(
            mo.md(
                f"""
                **Checkpoint not found:** `{_checkpoint}`

                Download the submitted checkpoint from the
                [GitHub release]({RELEASE_URL}), then set its path above.
                """
            ),
            kind="warn",
        ),
    )
    selected_checkpoint = _checkpoint
    return (selected_checkpoint,)


@app.cell(hide_code=True)
def _(
    CoarseVorticityStepper,
    DNSConfig,
    EXPECTED_SHA256,
    SolverConfig,
    closure_rollout,
    generate_reference_trajectory,
    jax,
    load_candidate_params_with_digest,
    mo,
    no_closure_rollout,
    np,
    selected_checkpoint,
    validation_seed,
):
    _expected = EXPECTED_SHA256
    with mo.status.spinner(title="Loading checkpoint and generating rollouts"):
        _checkpoint, _params, checkpoint_digest = load_candidate_params_with_digest(
            selected_checkpoint,
            expected_sha256=_expected,
        )
        solver_config = SolverConfig(**_checkpoint["solver_config"])
        dns_config = DNSConfig(**_checkpoint["dns_config"])
        reference = generate_reference_trajectory(
            seed=int(validation_seed.value),
            num_steps=30,
            split="validation",
            config=dns_config,
        )
        _stepper = CoarseVorticityStepper(solver_config)
        learned_trajectory = np.asarray(
            jax.device_get(
                closure_rollout(_stepper, _params, reference.initial_coarse, 30)
            )
        )
        baseline_trajectory = np.asarray(
            jax.device_get(no_closure_rollout(_stepper, reference.initial_coarse, 30))
        )
        reference_trajectory = np.asarray(jax.device_get(reference.targets))

    completed_updates = int(_checkpoint["completed_updates"])
    return (
        baseline_trajectory,
        checkpoint_digest,
        completed_updates,
        learned_trajectory,
        reference_trajectory,
    )


@app.cell(hide_code=True)
def _(
    baseline_trajectory,
    checkpoint_digest,
    completed_updates,
    learned_trajectory,
    mo,
    np,
    reference_trajectory,
    validation_seed,
):
    learned_step_mse = np.mean(
        (learned_trajectory - reference_trajectory) ** 2, axis=(1, 2, 3)
    )
    baseline_step_mse = np.mean(
        (baseline_trajectory - reference_trajectory) ** 2, axis=(1, 2, 3)
    )
    learned_rollout_mse = float(np.mean(learned_step_mse))
    baseline_rollout_mse = float(np.mean(baseline_step_mse))
    relative_reduction = 1.0 - learned_rollout_mse / baseline_rollout_mse

    mo.vstack(
        [
            mo.callout(
                f"Verified checkpoint `{checkpoint_digest[:12]}…`, evaluated on "
                f"validation seed {validation_seed.value}.",
                kind="success",
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{learned_rollout_mse:.4f}",
                        label="Learned rollout MSE",
                        caption="30-step prefix",
                    ),
                    mo.stat(
                        value=f"{baseline_rollout_mse:.4f}",
                        label="No-closure MSE",
                        caption="same initial condition",
                    ),
                    mo.stat(
                        value=f"{100 * relative_reduction:.1f}%",
                        label="Error reduction",
                        caption="against no closure",
                        direction="increase",
                    ),
                    mo.stat(
                        value=f"{completed_updates}",
                        label="Training updates",
                        caption="checkpoint provenance",
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
        ],
        gap=0.75,
    )
    return baseline_step_mse, learned_step_mse


@app.cell(hide_code=True)
def _(mo):
    frame_step = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=30,
        label="Inspect rollout step",
        show_value=True,
        full_width=True,
    )
    mo.vstack(
        [mo.md("## 3. Watch the closure correct the coarse dynamics"), frame_step]
    )
    return (frame_step,)


@app.cell(hide_code=True)
def _(np, plt):
    _COLOURS = {
        "blue": "#0072B2",
        "green": "#009E73",
        "orange": "#E69F00",
        "graphite": "#222222",
    }

    def plot_fields(reference, learned, baseline, step):
        _index = int(step) - 1
        _truth = reference[_index, 0]
        _learned = learned[_index, 0]
        _baseline = baseline[_index, 0]
        _limit = float(np.max(np.abs(_truth)))
        _error_limit = float(
            max(
                np.max(np.abs(_learned - _truth)),
                np.max(np.abs(_baseline - _truth)),
            )
        )
        _fig, _axes = plt.subplots(2, 3, figsize=(12.2, 7.0), constrained_layout=True)
        _fields = (_truth, _learned, _baseline)
        _titles = ("Filtered DNS", "Learned closure", "No closure")
        for _axis, _field, _title in zip(_axes[0], _fields, _titles, strict=True):
            _image = _axis.imshow(
                _field,
                cmap="RdBu_r",
                vmin=-_limit,
                vmax=_limit,
                origin="lower",
                interpolation="nearest",
            )
            _axis.set_title(_title)
            _axis.set_xticks([])
            _axis.set_yticks([])
        _fig.colorbar(_image, ax=_axes[0], shrink=0.75, label="Vorticity ω")

        _axes[1, 0].axis("off")
        for _axis, _error, _title in (
            (_axes[1, 1], _learned - _truth, "Learned error"),
            (_axes[1, 2], _baseline - _truth, "No-closure error"),
        ):
            _error_image = _axis.imshow(
                _error,
                cmap="PuOr_r",
                vmin=-_error_limit,
                vmax=_error_limit,
                origin="lower",
                interpolation="nearest",
            )
            _axis.set_title(_title)
            _axis.set_xticks([])
            _axis.set_yticks([])
        _fig.colorbar(
            _error_image,
            ax=_axes[1, 1:],
            shrink=0.75,
            label="Prediction − filtered DNS",
        )
        _fig.suptitle(f"Validation rollout at step {step}", fontweight="bold")
        return _fig

    def plot_mse(learned_mse, baseline_mse, step):
        _steps = np.arange(1, len(learned_mse) + 1)
        _fig, _axis = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)
        _axis.plot(
            _steps,
            learned_mse,
            color=_COLOURS["blue"],
            linewidth=2.0,
            label="Learned closure",
        )
        _axis.plot(
            _steps,
            baseline_mse,
            color=_COLOURS["green"],
            linestyle="-.",
            linewidth=1.8,
            label="No closure",
        )
        _axis.axvline(step, color="0.55", linestyle=":", linewidth=1.2)
        _axis.scatter(
            [step, step],
            [learned_mse[step - 1], baseline_mse[step - 1]],
            color=[_COLOURS["blue"], _COLOURS["green"]],
            zorder=3,
        )
        _axis.set(
            xlabel="Rollout step",
            ylabel="Per-step vorticity MSE",
            yscale="log",
            xlim=(1, len(_steps)),
        )
        _axis.grid(axis="y", alpha=0.2)
        _axis.legend(frameon=False)
        return _fig

    return plot_fields, plot_mse


@app.cell(hide_code=True)
def _(
    baseline_step_mse,
    baseline_trajectory,
    frame_step,
    learned_step_mse,
    learned_trajectory,
    mo,
    plot_fields,
    plot_mse,
    reference_trajectory,
):
    field_figure = plot_fields(
        reference_trajectory,
        learned_trajectory,
        baseline_trajectory,
        frame_step.value,
    )
    mse_figure = plot_mse(
        learned_step_mse,
        baseline_step_mse,
        frame_step.value,
    )
    mo.vstack([field_figure, mse_figure], gap=1.25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Inspect what is preserved in Fourier space

    Vorticity-MSE is the training objective. Energy and enstrophy spectra are
    independent diagnostics. They test whether the learned trajectory retains
    physically important scale-by-scale structure rather than merely reducing a
    pixel-space error.
    """)
    return


@app.cell(hide_code=True)
def _(
    baseline_trajectory,
    coarse_grid_max_wavenumber,
    frame_step,
    learned_trajectory,
    mo,
    plt,
    radial_spectra,
    reference_trajectory,
    spectral_distance,
):
    _index = frame_step.value - 1
    _cutoff = coarse_grid_max_wavenumber()
    _spectra = {
        "Filtered DNS": radial_spectra(reference_trajectory[_index]).truncated(_cutoff),
        "Learned closure": radial_spectra(learned_trajectory[_index]).truncated(
            _cutoff
        ),
        "No closure": radial_spectra(baseline_trajectory[_index]).truncated(_cutoff),
    }
    _styles = {
        "Filtered DNS": ("#000000", "-", 2.2),
        "Learned closure": ("#0072B2", "-", 1.8),
        "No closure": ("#009E73", "-.", 1.8),
    }
    spectra_figure, _axes = plt.subplots(
        1, 2, figsize=(10.8, 4.0), constrained_layout=True
    )
    for _name, _spectrum in _spectra.items():
        _colour, _linestyle, _width = _styles[_name]
        _axes[0].loglog(
            _spectrum.wavenumber,
            _spectrum.energy,
            color=_colour,
            linestyle=_linestyle,
            linewidth=_width,
            label=_name,
        )
        _axes[1].loglog(
            _spectrum.wavenumber,
            _spectrum.enstrophy,
            color=_colour,
            linestyle=_linestyle,
            linewidth=_width,
            label=_name,
        )
    for _axis, _title, _ylabel in (
        (_axes[0], "Energy spectrum", "E(k)"),
        (_axes[1], "Enstrophy spectrum", "Z(k)"),
    ):
        _axis.set(title=_title, xlabel="Wavenumber k", ylabel=_ylabel)
        _axis.grid(alpha=0.2, which="both")
    _axes[0].legend(frameon=False)
    spectra_figure.suptitle(
        f"Radial spectra at step {frame_step.value}", fontweight="bold"
    )

    learned_spectral_distance = spectral_distance(
        _spectra["Learned closure"].energy,
        _spectra["Filtered DNS"].energy,
    )
    baseline_spectral_distance = spectral_distance(
        _spectra["No closure"].energy,
        _spectra["Filtered DNS"].energy,
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        value=f"{learned_spectral_distance:.3f}",
                        label="Learned energy-spectrum distance",
                    ),
                    mo.stat(
                        value=f"{baseline_spectral_distance:.3f}",
                        label="No-closure energy-spectrum distance",
                    ),
                ],
                justify="start",
                gap=1.0,
            ),
            spectra_figure,
        ],
        gap=0.75,
    )
    return


@app.cell(hide_code=True)
def _(REPO_ROOT, json):
    with (REPO_ROOT / "docs" / "results" / "container-optimiser-demo.json").open(
        encoding="utf-8"
    ) as _handle:
        gradient_evidence = json.load(_handle)
    with (REPO_ROOT / "docs" / "results" / "final-metrics.json").open(
        encoding="utf-8"
    ) as _handle:
        selection_evidence = json.load(_handle)
    return gradient_evidence, selection_evidence


@app.cell(hide_code=True)
def _(gradient_evidence, mo, selection_evidence):
    _selected = selection_evidence["selection"]["candidates"]["aposteriori-700"]
    mo.vstack(
        [
            mo.md("## 5. Put one trajectory in context"),
            mo.md(
                "The interactive plots above show one seed. The locked selection "
                "protocol averages over all **32 validation seeds**. Separately, the "
                "served two-container demo verifies the reverse path itself."
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{_selected['mean_vorticity_mse']:.4f}",
                        label="32-seed validation MSE",
                        caption="selected at horizon 30",
                    ),
                    mo.stat(
                        value=f"{gradient_evidence['gradient_size']:,}",
                        label="Served gradient entries",
                        caption="all finite",
                    ),
                    mo.stat(
                        value=f"{100 * gradient_evidence['solver_transpose_sensitivity']:.2f}%",
                        label="Solver-transpose sensitivity",
                        caption="two-step parameter gradient",
                    ),
                    mo.stat(
                        value=(
                            f"{gradient_evidence['solver_vjp_calls']} + "
                            f"{gradient_evidence['closure_vjp_calls']}"
                        ),
                        label="Served VJP calls",
                        caption="solver + closure",
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.callout(
                "The submitted model was trained through the equivalent in-process "
                "composition for throughput. The served-image run demonstrates the "
                "deployment boundary; it did not produce the submitted parameters.",
                kind="neutral",
            ),
        ],
        gap=0.75,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **The closure is solver-aware.** It is trained on rollout error, not only
       an instantaneous tendency target.
    2. **The AD boundary is explicit.** JAX owns the rollout tape while PyTorch
       supplies forward and VJP callbacks for every CNN parameter.
    3. **The comparison is controlled.** Filtered DNS, learned closure, and the
       under-resolved baseline share the same initial condition and locked
       numerical configuration.
    4. **The claims stay bounded.** Results cover one freely decaying 2D regime;
       conservation, forced turbulence, and 3D transfer remain open.

    Continue with [`README.md`](../README.md) for the full evaluation table and
    [`docs/VERIFICATION.md`](../docs/VERIFICATION.md) for the measured evidence.
    """)
    return


if __name__ == "__main__":
    app.run()
