# Verification record

Completed runs for this submission, with the measured values. Every
number below was produced locally on the working tree it describes. Anything not
listed here is not claimed.

The headline scientific results, all per-seed errors, and a SHA-256 integrity
manifest over the preserved source reports and checkpoints are in
[`results/final-metrics.json`](results/final-metrics.json).

## Locked configuration

- On-demand `256²` Exponax DNS, sharply filtered in Fourier space to `64²` coarse truth.
- Mean-zero Gaussian random-field initial conditions, radial Fourier support `10 ≤ |k| ≤ 32`, rescaled to `|ω|ₘₐₓ = 20`.
- JAX/Exponax coarse vorticity solver, ETDRK2, float32, `ν = 10⁻³`, `dt = 0.002`.
- PyTorch scalar vorticity-tendency CNN: ten layers, 64 channels, `5×5` periodic kernels, 822,977 flattened parameters, fixed seed-zero initialisation.
- Operator split: Exponax base step, then `dt × closure(base_state)`.
- A-posteriori rollout vorticity-MSE, rematerialised `1 → 5 → 30` unroll curriculum, Adam at `10⁻⁴`.
- Disjoint deterministic seed splits: training `0–9999`, validation `10000–10031`, test `20000–20031`.
- Baselines: matched a-priori CNN, no closure, clipped dynamic Smagorinsky, static Smagorinsky with `Cₛ = 0.17`.

These knobs are enforced by the config dataclasses
(`src/tesseract_hybrid_closure/configs.py`).

## Model selection

The final checkpoint was produced by continuing the intact 300-update
checkpoint for 400 further unroll-30 updates on training seeds `300–699`
(16.28 minutes; continuation loss `1.7387e-2 → 5.5547e-3`; all parameters and
stored losses finite; source checkpoint unchanged).

Selection used all 32 validation seeds at horizon 30, recorded before any test
access:

| Candidate | Updates | Mean vorticity MSE | Population std |
|---|---:|---:|---:|
| a-posteriori 300 | 300 | 0.01468837 | 0.00409362 |
| **a-posteriori 700** | 700 | **0.00810003** | **0.00197358** |

The 700-update checkpoint was locked, at 44.9% lower validation MSE.

## Sealed test evaluation

Seeds `20000–20031`, 32 trajectories, rolled once to 500 steps per seed with
shorter horizons taken as prefixes. Both learned models use 700 Adam updates and
share architecture, initialisation, learning rate, training-seed budget, solver
configuration, and DNS configuration. All 32 trajectories of every method
remained finite through 500 steps.

| Method | 30 | 60 | 120 | 250 | 500 |
|---|---:|---:|---:|---:|---:|
| **A-posteriori** | **0.00728045** | **0.0274173** | **0.108572** | **0.445800** | **1.146774** |
| A-priori, matched | 0.00762237 | 0.0298035 | 0.123548 | 0.523401 | 1.360247 |
| No closure | 0.0777558 | 0.277713 | 0.904837 | 2.462234 | 4.051578 |
| Dynamic Smagorinsky | 0.0779627 | 0.272349 | 0.854092 | 2.190985 | 3.394246 |
| Static Smagorinsky, `Cₛ=0.17` | 0.0880588 | 0.295550 | 0.868222 | 2.036504 | 2.944275 |

The a-posteriori model beats the matched a-priori model on all 32 seeds at every
reported horizon, reduces mean MSE against no closure by 90.6% at 30 steps and
71.7% at 500, and against matched a-priori by 4.5% at 30 steps and 15.7% at 500.

The sealed test split was opened once and must not be reused for model choice.

## Spectral diagnostic

Radially binned energy and enstrophy spectra over all 32 validation seeds, at
rollout steps 30, 120 and 500, for five methods against the filtered-DNS
reference. The driver refuses any split but validation, and refuses to compare
checkpoints trained in different regimes. Summary statistic: mean absolute
log-ratio to the reference spectrum over shells `1 <= k <= 32`.

| Method | 30 | 120 | 500 |
|---|---:|---:|---:|
| A-posteriori | **0.0693** | **0.1187** | 0.3019 |
| A-priori, matched | 0.0944 | 0.1530 | 0.3699 |
| No closure | 0.0725 | 0.1330 | 0.3771 |
| Dynamic Smagorinsky | 0.0874 | 0.1330 | **0.2787** |
| Static Smagorinsky, `Cs=0.17` | 0.1280 | 0.2499 | 0.5480 |

Enstrophy log-distance. The a-posteriori model is spectrally closer than the
matched a-priori model at every step, and closest of all five methods at steps
30 and 120. At step 500 dynamic Smagorinsky is spectrally closest despite a
roughly threefold worse vorticity MSE, so pointwise accuracy and spectral
fidelity separate at long rollout. Full spectra and the energy log-distances are
in `docs/results/spectra-validation.json`.

Diagnostic only: no reported metric, model selection, or test-split result
depends on it.

### Mode-wise amplitude/phase decomposition

At each requested step, the same sweep splits the endpoint field's squared
vorticity error mode by mode using
`|a-b|^2 = (|a|-|b|)^2 + 2|a||b|(1-cos(arg a - arg b))`, which is exact and
non-negative term by term. Summing every shell returns the endpoint vorticity
MSE at that step. It does not return the reported rollout-prefix MSE, which
averages every state up to that step. A test pins the endpoint identity against
an independently computed `vorticity_mse` to 1e-5 relative. Totals at step 500,
32 validation seeds:

| Method | Amplitude | Phase | Phase share |
|---|---:|---:|---:|
| A-posteriori | **1.2297** | **1.1461** | 0.482 |
| A-priori, matched | 1.5291 | 1.3039 | 0.460 |
| No closure | 2.1468 | 3.6745 | 0.631 |
| Dynamic Smagorinsky | 1.6129 | 2.9436 | 0.646 |
| Static Smagorinsky | 1.5349 | 2.1941 | 0.588 |

The a-posteriori model has the lowest endpoint amplitude and phase error of all
five methods at every measured step. At step 500, its phase error is 61.1%
lower than dynamic Smagorinsky's and its amplitude error is 23.8% lower. Against
the matched a-priori model, the differences are 12.1% and 19.6%, respectively.
The growing phase share for the classical baselines is consistent with greater
endpoint phase misalignment; it does not identify a dynamical mechanism.

Dynamic Smagorinsky's lower step-500 spectral log-distance is compatible with
its larger absolute amplitude and phase errors because the diagnostics weight
modes differently. Spectral log-distance gives each shell equal log-scale
weight, whereas the decomposition totals weight squared coefficient errors and
are dominated by energetic modes.

Framing after Koehler, *From Numerical Simulators of PDEs to Neural Emulators
and Back* (2026), arXiv:2608.24547. The closed-form Fourier-multiplier analysis
there applies to schemes that diagonalise in Fourier space; this system is
nonlinear in both the advection and the closure, so this is the empirical
counterpart.

## Served two-image evidence

Two separately built images are composed with `tesseract-jax`:

- `coarse_solver:0.2.0`, JAX/Exponax: apply, JVP, VJP, abstract evaluation;
- `scalar_closure:0.2.0`, PyTorch: apply and VJP over the input field and flattened parameters.

One accepted Adam update on a **two-step** rollout objective in the calibrated
regime, with client-boundary instrumentation on every endpoint call and a demo
that fails closed if the solver VJP evidence is absent:

- seed `0`, vorticity amplitude `20`, timestep `0.002`, unrolled steps `2`;
- objective: two-step composed rollout vorticity-MSE against both filtered DNS targets;
- loss `0.0008655003 → 0.0008654960`;
- gradient: all 822,977 entries finite; parameter update norm `0.00462866`;
- measured served endpoint calls: solver apply 6, solver VJP 1 (input `omega`, min cotangent norm `5.812e-4`), closure apply 6, closure VJP 2 (`omega`+`params_flat`, then `params_flat`);
- accepted updates: `1`.

Because the objective unrolls for two steps, the parameter gradient necessarily
carries a cotangent through the second solver step. A sensitivity test zeroes the
solver VJP's `omega` cotangent at the endpoint boundary and observes
`||g_true − g_zero|| / ||g_true|| = 0.3998` on the same served clients
(`docs/results/container-optimiser-demo.json`), pinning that the solver
transpose contributes materially to the parameter gradient; the identical
measurement on the local public-API clients is pinned in
`tests/test_tesseract_composition.py`.

Full a-posteriori training uses the equivalent in-process PyTorch callback and
explicit VJP for throughput. It does not call the served containers at every
training step.

## Training through the served containers

A-posteriori training in which every forward evaluation and every gradient
crossed the two served images over HTTP. The tracked record
(`docs/results/served-training.json`) is the final run of 2026-08-30 (500 Adam
updates, one fresh train-split trajectory per update, two-step rollout
objective, from the same fixed seed-zero initialisation as the reported run):

- training objective, first to last update: `8.655e-4` to `1.005e-4`;
- 30-step validation MSE on held-out seeds `10000-10007`: `0.085425` to `0.009446`, an 88.9% reduction;
- the submitted in-process model scores `0.008248` on those same eight seeds, scored identically;
- measured served calls: solver apply 1000, solver VJP 500, closure apply 1000, closure VJP 1000;
- all 822,977 gradient entries finite at every update; the run aborts rather than continuing on a non-finite gradient;
- wall clock 285 s; composition invariants checked at the end and the run fails closed if the VJP evidence is absent;
- provenance per component: name/version from the built config, config-file SHA-256, served OpenAPI schema digest, and the live Docker image ID (solver `sha256:f844b829…`, closure `sha256:91e8a522…`);
- source manifest: git commit `c5956a2`, branch `main`, clean tree (dirty state and dirty files recorded in general);
- the trained parameters are checkpointed atomically and their SHA-256 recorded (`55410417…`).

An earlier identical-protocol run (commit `54e9fd8`) gave `0.009414`; repeats
reproduce to roughly four significant figures rather than bit-exactly
(`0.009414`, `0.009446`, `0.009459` observed), which is ordinary float32
non-determinism across the served boundary.

This is a demonstration of the deployment boundary, not the submitted model. It
uses a fixed two-step unroll rather than the `1 -> 5 -> 30` curriculum, touches
no test seed, and no reported number depends on it.

## The headline unroll through the served containers

100 Adam updates at the locked 30-step unroll, the final curriculum stage's
length, with every forward and reverse step crossing the two images
(`docs/results/served-training-30unroll.json`):

- 30-step validation MSE on seeds `10000-10007`: `0.085425` to `0.036354`, a 57.4% reduction;
- measured served calls: solver VJP 2900, closure VJP 3000 (30 per update, with the first-step state cotangent pruned as in the two-step demo);
- all 822,977 gradient entries finite at every update; wall clock 771 s;
- provenance bound to commit `e46c0c3` (clean tree), image IDs `f844b829…` and `91e8a522…`, checkpoint SHA-256 `42f15921…`;
- the 700-update in-process reference scores `0.008248` on the same seeds, so the run shows the headline objective trains through the served boundary, not that 100 served updates match the full curriculum.

## Solver-transpose sensitivity through the served images

The `0.3998` relative sensitivity, the served loss pair and the endpoint call
counts are all recorded under Served two-image evidence above, with the
image-backed report in `docs/results/container-optimiser-demo.json` and the
local pin in `tests/test_tesseract_composition.py`.

## Checkpoint distribution

The exact submitted and a-priori parameters are published as GitHub release
`submission-2026-08-31` (SHA-256 `8ed5b36f…` and `8862fbd5…`), so a clean clone
can evaluate the reported model without retraining.

## Defects found by review and live testing, and fixed

- The one-step served demo did not prove reverse mode through the served solver Tesseract: with a fixed initial state, the parameter gradient needs only the closure VJP. Replaced by the two-step rollout objective above, with call instrumentation and the solver-VJP sensitivity test.

## Provenance hardening

- Selection records the SHA-256 of every candidate checkpoint as loaded. `--stage evaluate --split test` rejects missing or mismatched digests before any trajectory is generated, and the matched a-priori checkpoint digest is recorded in the evaluation evidence.
- `docs/results/final-metrics.json` (schema v2) embeds all 32×5×5 per-seed errors, so every aggregate and paired win count is recomputable from the tracked file alone, plus a post-evaluation integrity manifest of SHA-256 digests and byte sizes over the preserved selection report, evaluation report, candidate checkpoints, selected checkpoint, a-priori checkpoint, and a-priori training summary. The manifest is post-evaluation evidence; it is not proof of pre-test locking.

## Checks run on the current tree

- Ruff checks and format verification.
- The full non-smoke unit suite passed, including capability probes, two-step instrumentation, the solver-VJP sensitivity test, provenance-digest tests, and schema-v2 tests.
- The single-step cross-framework gradient smoke and both rollout smokes (30-step rollout gradient, one Adam update per curriculum stage).
- Spectral-diagnostic invariants: Parseval (shell-summed enstrophy equals `0.5 * mean(omega**2)` to `rtol=1e-12`), single-mode shell localisation against its analytic value, the exact per-mode relation `Z_q = |q|**2 E_q`, and the band-limited initial condition carrying no enstrophy outside `10 <= k <= 32`. Rounded radial shells conserve their summed quantities but do not generally satisfy `Z(k) = k**2 E(k)` exactly because a shell contains modes with different `|q|`.
- Both Tesseract images rebuilt from the working tree, each staging `LICENSE`, `NOTICE` and `ATTRIBUTION.md` under `/tesseract/licenses/` (verified inside the images); container composition smoke passed on the two-step objective, asserting measured solver VJP (≥1) and closure VJP (≥2) calls and a finite non-zero 822,977-parameter gradient.
- The image-backed Adam demonstration reproduced the served numbers above, extended with a solver-transpose sensitivity of `0.3998` measured through the served clients (`docs/results/container-optimiser-demo.json`).
- The exact submitted and a-priori checkpoints are published as release `submission-2026-08-31` with recorded SHA-256 digests.
- The curated metrics regenerated as schema v2 with unchanged headline values; all committed figures and the animation byte-identical.

`make verify` reproduces the non-Docker portion of this list from a clean clone,
and runs in CI on every push.
