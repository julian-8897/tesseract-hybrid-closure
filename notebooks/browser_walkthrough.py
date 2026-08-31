# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.24.0",
#     "matplotlib==3.11.1",
#     "numpy==2.5.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Differentiable hybrid closure")


@app.cell(hide_code=True)
def _():
    import hashlib
    import json
    from io import BytesIO

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return BytesIO, hashlib, json, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        r"""
        <style>
          :root {
            --ink: #122033;
            --muted: #526177;
            --blue: #0072B2;
            --teal: #009E73;
            --orange: #E69F00;
            --paper: #ffffff;
            --wash: #f4f7fb;
            --line: #dbe3ed;
          }
          body { color: var(--ink); }
          .marimo { background: #fbfcfe; }
          .hero {
            background: linear-gradient(135deg, #0b1f36 0%, #123b5d 62%, #076b72 100%);
            color: white;
            border-radius: 18px;
            padding: 2.4rem 2.7rem 2.2rem;
            margin: 0.2rem 0 1.4rem;
            box-shadow: 0 18px 45px rgba(18, 32, 51, 0.16);
          }
          .hero .eyebrow {
            color: #8de0db;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
          }
          .hero h1 {
            color: white;
            font-family: var(--marimo-heading-font, ui-serif, Georgia, serif);
            font-size: clamp(2.15rem, 5vw, 3.8rem);
            line-height: 1.04;
            margin: 0;
            max-width: 820px;
          }
          .hero p {
            color: #e4edf6;
            font-size: 1.08rem;
            line-height: 1.65;
            max-width: 850px;
            margin: 1.1rem 0 0;
          }
          .hero .tags { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1.35rem; }
          .hero .tag {
            background: rgba(255,255,255,0.11);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 999px;
            color: white;
            font-size: 0.82rem;
            padding: 0.35rem 0.7rem;
          }
          .section-kicker {
            color: var(--blue);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
          }
          .flow-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr auto 1fr;
            align-items: stretch;
            gap: 0.7rem;
            margin: 1rem 0 0.6rem;
          }
          .flow-card {
            background: white;
            border: 1px solid var(--line);
            border-radius: 13px;
            padding: 1rem 1.05rem;
            min-height: 108px;
            box-shadow: 0 7px 20px rgba(18, 32, 51, 0.06);
          }
          .flow-card.blue { border-top: 4px solid var(--blue); }
          .flow-card.teal { border-top: 4px solid var(--teal); }
          .flow-card.orange { border-top: 4px solid var(--orange); }
          .flow-card strong { display: block; font-size: 1rem; margin-bottom: 0.35rem; }
          .flow-card span { color: var(--muted); font-size: 0.88rem; line-height: 1.4; }
          .arrow { align-self: center; color: #718096; font-size: 1.55rem; }
          .reverse-strip {
            background: #eef6fb;
            border: 1px solid #cce2f0;
            border-radius: 10px;
            color: #174d70;
            font-size: 0.88rem;
            padding: 0.65rem 0.9rem;
            text-align: center;
          }
          .explorer-shell {
            background: white;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 10px 30px rgba(18, 32, 51, 0.07);
          }
          .metric-note { color: var(--muted); font-size: 0.9rem; line-height: 1.5; }
          @media (max-width: 760px) {
            .hero { padding: 1.6rem 1.3rem; }
            .flow-grid { grid-template-columns: 1fr; }
            .arrow { transform: rotate(90deg); text-align: center; }
          }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
async def _(BytesIO, hashlib, json, mo, np):
    _location = mo.notebook_location()
    if _location is None:
        raise RuntimeError("the notebook location is unavailable")
    _root = str(_location).rstrip("/") + "/public/"
    _is_remote = _root.startswith(("http://", "https://"))

    async def _read_bytes(filename):
        if _is_remote:
            from pyodide.http import pyfetch

            _response = await pyfetch(_root + filename)
            if _response.status != 200:
                raise RuntimeError(
                    f"failed to load {filename}: HTTP {_response.status}"
                )
            return await _response.bytes()
        from pathlib import Path

        return (Path(_location) / "public" / filename).read_bytes()

    _metadata_bytes = await _read_bytes("rollout_seed10000.json")
    demo_metadata = json.loads(_metadata_bytes.decode("utf-8"))

    _arrays = {}
    for _label, _record in demo_metadata["arrays"].items():
        _payload = await _read_bytes(_record["path"])
        _digest = hashlib.sha256(_payload).hexdigest()
        if _digest != _record["sha256"]:
            raise RuntimeError(f"integrity check failed for {_record['path']}")
        if _is_remote:
            _temporary = f"/tmp/{_record['path']}"
            with open(_temporary, "wb") as _handle:
                _handle.write(_payload)
            _arrays[_label] = np.load(_temporary, allow_pickle=False)
        else:
            _arrays[_label] = np.load(BytesIO(_payload), allow_pickle=False)

    filtered_dns = _arrays["filtered_dns"]
    learned_closure = _arrays["learned_closure"]
    no_closure = _arrays["no_closure"]
    return demo_metadata, filtered_dns, learned_closure, no_closure


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        r"""
        <div class="hero">
          <div class="eyebrow">Differentiable scientific machine learning</div>
          <h1>Learning a closure through the dynamics it changes</h1>
          <p>
            A PyTorch CNN corrects an under-resolved JAX/Exponax spectral solver
            for freely decaying two-dimensional turbulence. The model is trained
            through complete solver rollouts, with explicit VJPs carrying gradients
            across the framework boundary.
          </p>
          <div class="tags">
            <span class="tag">PyTorch closure</span>
            <span class="tag">JAX spectral solver</span>
            <span class="tag">A-posteriori training</span>
            <span class="tag">Cross-framework reverse mode</span>
          </div>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div class="section-kicker">The modelling problem</div>

    ## What is missing on a coarse grid?

    The 256² reference resolves vorticity modes that a 64² simulation cannot
    represent. Sharp filtering gives the coarse truth, but evolving that field
    with the coarse solver alone omits the net influence of discarded scales.
    The learned closure supplies a scalar vorticity tendency that compensates for
    this unresolved feedback.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        r"""
        <div class="flow-grid">
          <div class="flow-card orange">
            <strong>Filtered DNS target</strong>
            <span>256² reference trajectory sharply filtered to the 64² resolved modes.</span>
          </div>
          <div class="arrow">→</div>
          <div class="flow-card blue">
            <strong>Mechanistic step</strong>
            <span>JAX/Exponax advances coarse vorticity with float32 ETDRK2.</span>
          </div>
          <div class="arrow">→</div>
          <div class="flow-card teal">
            <strong>Learned correction</strong>
            <span>PyTorch CNN predicts qθ(ω*), then ωₙ₊₁ = ω* + Δt qθ.</span>
          </div>
        </div>
        <div class="reverse-strip">
          rollout loss ⟶ closure VJP ⟶ solver VJP ⟶ all 822,977 CNN parameters
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    explanation_tabs = mo.ui.tabs(
        {
            "Why a-posteriori?": mo.md(r"""
                **The objective sees trajectories.** A closure prediction at an early
                step changes every later solver state. Minimising rollout MSE therefore
                trains the CNN for its coupled effect on resolved dynamics, rather than
                fitting an instantaneous tendency in isolation.
            """),
            "How gradients cross": mo.md(r"""
                **JAX owns the rollout tape.** `jax.custom_vjp` wraps NumPy-boundary
                callbacks. PyTorch autograd returns the closure-input and parameter
                cotangents; `jax.vjp` propagates the state cotangent through Exponax.
                No finite-difference gradient or surrogate backward model is used.
            """),
            "What is held fixed?": mo.md(r"""
                **The comparison changes only the closure.** All trajectories use the
                same initial condition, 64² grid, ETDRK2 solver, Δt = 0.002,
                ν = 10⁻³ and filtered-DNS target. The browser data use validation seed
                10000, not the sealed test split.
            """),
        }
    )
    mo.vstack([explanation_tabs])
    return


@app.cell(hide_code=True)
def _(filtered_dns, learned_closure, no_closure, np):
    learned_step_mse = np.mean((learned_closure - filtered_dns) ** 2, axis=(1, 2, 3))
    baseline_step_mse = np.mean((no_closure - filtered_dns) ** 2, axis=(1, 2, 3))
    learned_prefix_mse = np.cumsum(learned_step_mse) / np.arange(1, 31)
    baseline_prefix_mse = np.cumsum(baseline_step_mse) / np.arange(1, 31)
    return (
        baseline_prefix_mse,
        baseline_step_mse,
        learned_prefix_mse,
        learned_step_mse,
    )


@app.cell(hide_code=True)
def _(mo):
    frame_step = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=30,
        label="Rollout step",
        show_value=True,
        include_input=True,
        full_width=True,
    )
    method_choice = mo.ui.radio(
        options=["Learned closure", "No closure"],
        value="Learned closure",
        label="Inspect method",
        inline=True,
    )
    field_view = mo.ui.dropdown(
        options=["Vorticity", "Signed error", "Absolute error"],
        value="Vorticity",
        label="Field view",
    )
    spectrum_choice = mo.ui.radio(
        options=["Energy", "Enstrophy"],
        value="Energy",
        label="Spectrum",
        inline=True,
    )
    controls = mo.vstack(
        [
            mo.Html('<div class="section-kicker">Interactive rollout explorer</div>'),
            mo.md("## Follow the coupled prediction through time"),
            frame_step,
            mo.hstack(
                [method_choice, field_view, spectrum_choice],
                align="end",
                gap=1.2,
            ),
        ],
        gap=0.8,
    )
    mo.vstack([controls])
    return field_view, frame_step, method_choice, spectrum_choice


@app.cell(hide_code=True)
def _(
    baseline_prefix_mse,
    baseline_step_mse,
    frame_step,
    learned_prefix_mse,
    learned_step_mse,
    method_choice,
    mo,
):
    _index = frame_step.value - 1
    _is_learned = method_choice.value == "Learned closure"
    _step_mse = learned_step_mse[_index] if _is_learned else baseline_step_mse[_index]
    _prefix_mse = (
        learned_prefix_mse[_index] if _is_learned else baseline_prefix_mse[_index]
    )
    _reference_mse = baseline_step_mse[_index]
    _step_reduction = 1.0 - learned_step_mse[_index] / _reference_mse
    _physical_time = frame_step.value * 0.002

    mo.hstack(
        [
            mo.stat(
                value=f"t = {_physical_time:.3f}",
                label="Physical time",
                caption=f"step {frame_step.value} of 30",
            ),
            mo.stat(
                value=f"{_step_mse:.4f}",
                label=f"{method_choice.value} step MSE",
                caption="current field",
            ),
            mo.stat(
                value=f"{_prefix_mse:.4f}",
                label="Rollout-prefix MSE",
                caption=f"steps 1–{frame_step.value}",
            ),
            mo.stat(
                value=f"{100 * _step_reduction:.1f}%",
                label="Learned error reduction",
                caption="at the selected step",
                direction="increase" if _step_reduction >= 0 else "decrease",
            ),
        ],
        widths="equal",
        gap=0.9,
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    _COLOURS = {
        "dns": "#111827",
        "learned": "#0072B2",
        "baseline": "#009E73",
        "accent": "#E69F00",
    }

    def _clean_axis(axis):
        axis.set_xticks([])
        axis.set_yticks([])
        for _spine in axis.spines.values():
            _spine.set_visible(False)

    def plot_field_explorer(reference, learned, baseline, step, method, view):
        _index = int(step) - 1
        _truth = reference[_index, 0]
        _learned = learned[_index, 0]
        _baseline = baseline[_index, 0]
        _selected = _learned if method == "Learned closure" else _baseline
        _other = _baseline if method == "Learned closure" else _learned
        _other_name = "No closure" if method == "Learned closure" else "Learned closure"

        _figure, _axes = plt.subplots(
            1, 3, figsize=(11.6, 3.75), constrained_layout=True
        )
        if view == "Vorticity":
            _limit = float(np.max(np.abs(_truth)))
            _panels = (
                (_truth, "Filtered DNS", "RdBu_r", -_limit, _limit),
                (_selected, method, "RdBu_r", -_limit, _limit),
                (_selected - _truth, f"{method} − DNS", "PuOr_r", None, None),
            )
        elif view == "Signed error":
            _errors = (_learned - _truth, _baseline - _truth)
            _limit = float(max(np.max(np.abs(_error)) for _error in _errors))
            _panels = (
                (_selected - _truth, f"{method} error", "PuOr_r", -_limit, _limit),
                (_other - _truth, f"{_other_name} error", "PuOr_r", -_limit, _limit),
                (
                    np.abs(_baseline - _truth) - np.abs(_learned - _truth),
                    "Where the closure helps",
                    "PiYG",
                    None,
                    None,
                ),
            )
        else:
            _errors = (np.abs(_learned - _truth), np.abs(_baseline - _truth))
            _limit = float(max(np.max(_error) for _error in _errors))
            _panels = (
                (np.abs(_selected - _truth), f"|{method} − DNS|", "magma", 0, _limit),
                (np.abs(_other - _truth), f"|{_other_name} − DNS|", "magma", 0, _limit),
                (
                    np.abs(_baseline - _truth) - np.abs(_learned - _truth),
                    "Positive: closure is better",
                    "PiYG",
                    None,
                    None,
                ),
            )

        for _axis, (_values, _title, _cmap, _vmin, _vmax) in zip(
            _axes, _panels, strict=True
        ):
            if _vmin is None or _vmax is None:
                _panel_limit = float(np.max(np.abs(_values)))
                _vmin, _vmax = -_panel_limit, _panel_limit
            _image = _axis.imshow(
                _values,
                cmap=_cmap,
                vmin=_vmin,
                vmax=_vmax,
                origin="lower",
                interpolation="nearest",
            )
            _axis.set_title(_title, fontsize=10.5, fontweight=600)
            _clean_axis(_axis)
            _figure.colorbar(_image, ax=_axis, shrink=0.72, pad=0.025)
        _figure.suptitle(
            f"Resolved vorticity at step {step}", fontsize=12, fontweight="bold"
        )
        return _figure

    def plot_error_growth(
        learned_step, baseline_step, learned_prefix, baseline_prefix, step
    ):
        _steps = np.arange(1, len(learned_step) + 1)
        _figure, _axes = plt.subplots(
            1, 2, figsize=(11.2, 3.55), constrained_layout=True
        )
        for _axis, _learned, _baseline, _title in (
            (_axes[0], learned_step, baseline_step, "Instantaneous field error"),
            (_axes[1], learned_prefix, baseline_prefix, "Rollout-prefix error"),
        ):
            _axis.plot(
                _steps,
                _learned,
                color=_COLOURS["learned"],
                linewidth=2.2,
                label="Learned closure",
            )
            _axis.plot(
                _steps,
                _baseline,
                color=_COLOURS["baseline"],
                linestyle="-.",
                linewidth=1.9,
                label="No closure",
            )
            _axis.axvline(step, color="0.55", linestyle=":", linewidth=1.1)
            _axis.scatter(
                [step, step],
                [_learned[step - 1], _baseline[step - 1]],
                color=[_COLOURS["learned"], _COLOURS["baseline"]],
                s=30,
                zorder=3,
            )
            _axis.set(
                title=_title,
                xlabel="Rollout step",
                ylabel="Vorticity MSE",
                yscale="log",
                xlim=(1, len(_steps)),
            )
            _axis.grid(axis="y", alpha=0.18)
        _axes[0].legend(frameon=False, loc="upper left")
        return _figure

    return plot_error_growth, plot_field_explorer


@app.cell(hide_code=True)
def _(np, plt):
    def radial_spectra(omega):
        _values = np.asarray(omega, dtype=np.float64)
        _num_points = _values.shape[-1]
        _kx = np.fft.fftfreq(_num_points) * _num_points
        _ky = np.fft.fftfreq(_num_points) * _num_points
        _squared = _kx[:, None] ** 2 + _ky[None, :] ** 2
        _shell = np.rint(np.sqrt(_squared)).astype(np.int64)
        _omega_hat = np.fft.fft2(_values) / _num_points**2
        _power = np.abs(_omega_hat) ** 2
        _enstrophy_density = 0.5 * _power
        with np.errstate(divide="ignore", invalid="ignore"):
            _energy_density = np.where(_squared > 0.0, 0.5 * _power / _squared, 0.0)
        _num_shells = int(_shell.max()) + 1
        _energy = np.bincount(
            _shell.ravel(), weights=_energy_density.ravel(), minlength=_num_shells
        )
        _enstrophy = np.bincount(
            _shell.ravel(),
            weights=_enstrophy_density.ravel(),
            minlength=_num_shells,
        )
        return np.arange(_num_shells), _energy, _enstrophy

    def plot_spectrum(reference, learned, baseline, step, quantity):
        _index = int(step) - 1
        _fields = {
            "Filtered DNS": reference[_index, 0],
            "Learned closure": learned[_index, 0],
            "No closure": baseline[_index, 0],
        }
        _styles = {
            "Filtered DNS": ("#111827", "-", 2.4),
            "Learned closure": ("#0072B2", "-", 2.0),
            "No closure": ("#009E73", "-.", 1.9),
        }
        _figure, (_main, _ratio) = plt.subplots(
            1,
            2,
            figsize=(11.2, 3.65),
            gridspec_kw={"width_ratios": [1.45, 1.0]},
            constrained_layout=True,
        )
        _computed = {}
        for _name, _field in _fields.items():
            _k, _energy, _enstrophy = radial_spectra(_field)
            _values = _energy if quantity == "Energy" else _enstrophy
            _mask = (_k >= 1) & (_k <= 32)
            _computed[_name] = (_k[_mask], _values[_mask])
            _colour, _linestyle, _width = _styles[_name]
            _main.loglog(
                _k[_mask],
                _values[_mask],
                color=_colour,
                linestyle=_linestyle,
                linewidth=_width,
                label=_name,
            )
        _reference_values = _computed["Filtered DNS"][1]
        for _name in ("Learned closure", "No closure"):
            _k, _values = _computed[_name]
            _colour, _linestyle, _width = _styles[_name]
            _ratio.semilogx(
                _k,
                _values / _reference_values,
                color=_colour,
                linestyle=_linestyle,
                linewidth=_width,
                label=_name,
            )
        _main.set(
            title=f"{quantity} spectrum",
            xlabel="Wavenumber k",
            ylabel="E(k)" if quantity == "Energy" else "Z(k)",
        )
        _ratio.axhline(1.0, color="0.35", linewidth=1.0, linestyle=":")
        _ratio.set(
            title="Ratio to filtered DNS",
            xlabel="Wavenumber k",
            ylabel="model / DNS",
        )
        _ratio.set_ylim(0.65, 1.35)
        for _axis in (_main, _ratio):
            _axis.grid(alpha=0.18, which="both")
        _main.legend(frameon=False)
        _figure.suptitle(
            f"Scale-by-scale agreement at step {step}", fontsize=12, fontweight="bold"
        )
        return _figure

    return (plot_spectrum,)


@app.cell(hide_code=True)
def _(
    baseline_prefix_mse,
    baseline_step_mse,
    field_view,
    filtered_dns,
    frame_step,
    learned_closure,
    learned_prefix_mse,
    learned_step_mse,
    method_choice,
    no_closure,
    plot_error_growth,
    plot_field_explorer,
    plot_spectrum,
    spectrum_choice,
):
    field_figure = plot_field_explorer(
        filtered_dns,
        learned_closure,
        no_closure,
        frame_step.value,
        method_choice.value,
        field_view.value,
    )
    error_figure = plot_error_growth(
        learned_step_mse,
        baseline_step_mse,
        learned_prefix_mse,
        baseline_prefix_mse,
        frame_step.value,
    )
    spectrum_figure = plot_spectrum(
        filtered_dns,
        learned_closure,
        no_closure,
        frame_step.value,
        spectrum_choice.value,
    )
    return error_figure, field_figure, spectrum_figure


@app.cell(hide_code=True)
def _(error_figure, field_figure, mo, spectrum_figure):
    explorer_tabs = mo.ui.tabs(
        {
            "Flow fields": field_figure,
            "Error through time": error_figure,
            "Resolved spectra": spectrum_figure,
        }
    )
    mo.vstack(
        [
            explorer_tabs,
            mo.md(
                "*Move the rollout slider or change a selector above. Every panel "
                "is recomputed in the browser from the precomputed validation fields.*"
            ),
        ],
        gap=0.6,
    )
    return


@app.cell(hide_code=True)
def _(demo_metadata, mo):
    provenance = mo.accordion(
        {
            "Data and checkpoint provenance": mo.md(
                f"""
                - **Split:** `{demo_metadata["split"]}`
                - **Seed:** `{demo_metadata["seed"]}`
                - **Checkpoint SHA-256:** `{demo_metadata["checkpoint_sha256"]}`
                - **Training updates:** `{demo_metadata["completed_updates"]}`
                - **DNS → coarse:** `{demo_metadata["dns_config"]["num_points"]}² → {demo_metadata["solver_config"]["num_points"]}²`
                - **Timestep:** `{demo_metadata["solver_config"]["dt"]}`

                The three browser arrays are SHA-256 checked before use. They are
                generated by `scripts/generate_notebook_demo_data.py` from the
                selected checkpoint. This is presentation data for one validation
                trajectory, not a new evaluation result.
                """
            ),
            "Scope and limitations": mo.md(r"""
                Evidence covers freely decaying two-dimensional turbulence in one
                calibrated regime. The scalar-tendency closure has no explicit
                conservation or dissipativity guarantee. Transfer to forced,
                stationary or three-dimensional turbulence is not tested.
            """),
        }
    )
    mo.vstack([provenance])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.Html('<div class="section-kicker">Measured evidence</div>'),
            mo.md("## The interactive case sits inside a broader evaluation"),
            mo.hstack(
                [
                    mo.stat(
                        value="71.7%",
                        label="Lower 500-step test MSE",
                        caption="relative to no closure",
                        direction="increase",
                    ),
                    mo.stat(
                        value="15.7%",
                        label="Lower 500-step test MSE",
                        caption="relative to matched a-priori CNN",
                        direction="increase",
                    ),
                    mo.stat(
                        value="32 / 32",
                        label="Paired test wins",
                        caption="all reported horizons",
                    ),
                    mo.stat(
                        value="39.98%",
                        label="Solver-transpose sensitivity",
                        caption="served two-step gradient",
                    ),
                ],
                widths="equal",
                gap=0.9,
            ),
            mo.callout(
                "The submitted parameters were trained through the equivalent "
                "in-process composition for throughput. A separate image-backed "
                "run verifies that the served solver and closure VJPs sustain "
                "genuine optimisation across the deployment boundary.",
                kind="neutral",
            ),
        ],
        gap=0.8,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this walkthrough establishes

    - **The physical comparison is inspectable.** The same filtered-DNS target,
      initial condition and numerical configuration are used throughout.
    - **The benefit develops through time.** The field, prefix-error and spectral
      views reveal different aspects of the coupled rollout.
    - **The gradient path is part of the method.** JAX and PyTorch retain their
      native implementations while explicit VJPs compose them into one objective.

    [Repository, reproducible live notebook and full verification record](https://github.com/julian-8897/tesseract-hybrid-closure)
    """)
    return


if __name__ == "__main__":
    app.run()
