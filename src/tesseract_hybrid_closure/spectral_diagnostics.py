"""Validation-split spectral diagnostic for the trained closures.

Answers a question the rollout vorticity-MSE cannot: *where in wavenumber* does
the learned closure change the flow, and does it hold the small scales at the
right amplitude rather than simply damping them? At each requested step it also
splits that step's endpoint-field squared vorticity error into an amplitude
part (getting the mode's magnitude wrong) and a phase part (getting its phase
wrong), so the small-scale result is attributable.

This is a diagnostic. It never touches the sealed test split, it changes no
reported number, and its own results are recorded separately from
``docs/results/final-metrics.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .baselines import smagorinsky_rollout
from .configs import DNSConfig, SolverConfig
from .data import generate_reference_trajectory
from .final_eval import load_candidate_params_with_digest
from .losses import closure_rollout, no_closure_rollout
from .solver import CoarseVorticityStepper
from .spectra import (
    coarse_grid_max_wavenumber,
    error_decomposition,
    radial_spectra,
    spectral_distance,
)

#: The diagnostic is confined to the validation split: the test split was
#: opened once and must stay sealed.
DIAGNOSTIC_SPLIT = "validation"

REFERENCE_METHOD = "filtered-dns"


def _configs_from_checkpoint(
    checkpoint: Mapping,
    checkpoint_path: Path,
) -> tuple[SolverConfig, DNSConfig]:
    solver_config = checkpoint.get("solver_config")
    dns_config = checkpoint.get("dns_config")
    if not solver_config or not dns_config:
        raise ValueError(
            f"{checkpoint_path} lacks persisted solver_config/dns_config; "
            "the diagnostic refuses to guess the regime"
        )
    return SolverConfig(**solver_config), DNSConfig(**dns_config)


def _validate_report_steps(report_steps: Sequence[int]) -> tuple[int, ...]:
    if not report_steps:
        raise ValueError("at least one report step is required")
    steps = tuple(int(step) for step in report_steps)
    if any(step <= 0 for step in steps):
        raise ValueError("report steps must be positive")
    if len(set(steps)) != len(steps):
        raise ValueError("report steps must be unique")
    return tuple(sorted(steps))


def build_methods(
    stepper: CoarseVorticityStepper,
    *,
    aposteriori_params: jax.Array,
    apriori_params: jax.Array | None,
    num_steps: int,
    include_smagorinsky: bool,
) -> dict[str, Callable[[jax.Array], jax.Array]]:
    """Rollout callables keyed by method name, all sharing one initial state."""
    methods: dict[str, Callable[[jax.Array], jax.Array]] = {
        "aposteriori": lambda state: closure_rollout(
            stepper, aposteriori_params, state, num_steps
        ),
        "no-closure": lambda state: no_closure_rollout(stepper, state, num_steps),
    }
    if apriori_params is not None:
        methods["apriori"] = lambda state: closure_rollout(
            stepper, apriori_params, state, num_steps
        )
    if include_smagorinsky:
        methods["dynamic-smagorinsky"] = lambda state: smagorinsky_rollout(
            stepper, state, num_steps, dynamic=True
        )
        methods["static-smagorinsky"] = lambda state: smagorinsky_rollout(
            stepper, state, num_steps, dynamic=False
        )
    return methods


def run_spectral_diagnostic(
    *,
    aposteriori_checkpoint: str | Path,
    apriori_checkpoint: str | Path | None,
    seeds: Sequence[int],
    report_steps: Sequence[int],
    split: str = DIAGNOSTIC_SPLIT,
    include_smagorinsky: bool = False,
) -> dict:
    """Seed-averaged energy and enstrophy spectra for every method.

    Each method is rolled once per seed to ``max(report_steps)`` from the same
    filtered-DNS initial state, and the requested steps are read off that one
    trajectory. Spectra are averaged over seeds shell by shell. At each
    requested step the report also carries the mode-wise amplitude/phase split
    of that step's endpoint-field vorticity MSE (the endpoint state against
    the filtered-DNS target at that step).
    """
    if split != DIAGNOSTIC_SPLIT:
        raise ValueError(
            f"the spectral diagnostic runs on the {DIAGNOSTIC_SPLIT!r} split only; "
            f"refusing {split!r} so the sealed test split stays sealed"
        )
    if not seeds:
        raise ValueError("at least one seed is required")
    requested_seeds = tuple(int(seed) for seed in seeds)
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("seeds must be unique")
    steps = _validate_report_steps(report_steps)
    num_steps = steps[-1]

    aposteriori_path = Path(aposteriori_checkpoint)
    checkpoint, aposteriori_params, aposteriori_digest = (
        load_candidate_params_with_digest(aposteriori_path)
    )
    solver_config, dns_config = _configs_from_checkpoint(checkpoint, aposteriori_path)

    apriori_params: jax.Array | None = None
    apriori_digest: str | None = None
    if apriori_checkpoint is not None:
        apriori_path = Path(apriori_checkpoint)
        apriori_checkpoint_data, apriori_params, apriori_digest = (
            load_candidate_params_with_digest(apriori_path)
        )
        apriori_solver, apriori_dns = _configs_from_checkpoint(
            apriori_checkpoint_data, apriori_path
        )
        if (apriori_solver, apriori_dns) != (solver_config, dns_config):
            raise ValueError(
                "the a-priori checkpoint was trained in a different regime than "
                "the a-posteriori checkpoint; refusing to compare their spectra"
            )

    stepper = CoarseVorticityStepper(solver_config)
    methods = build_methods(
        stepper,
        aposteriori_params=aposteriori_params,
        apriori_params=apriori_params,
        num_steps=num_steps,
        include_smagorinsky=include_smagorinsky,
    )
    max_wavenumber = coarse_grid_max_wavenumber(solver_config.num_points)

    names = (REFERENCE_METHOD, *methods)
    energy_sums: dict[str, dict[int, np.ndarray]] = {name: {} for name in names}
    enstrophy_sums: dict[str, dict[int, np.ndarray]] = {name: {} for name in names}
    amplitude_sums: dict[str, dict[int, np.ndarray]] = {name: {} for name in methods}
    phase_sums: dict[str, dict[int, np.ndarray]] = {name: {} for name in methods}
    wavenumber: np.ndarray | None = None

    def accumulate(name: str, step: int, field: np.ndarray) -> None:
        nonlocal wavenumber
        spectra = radial_spectra(
            field, domain_extent=solver_config.domain_extent
        ).truncated(max_wavenumber)
        if wavenumber is None:
            wavenumber = spectra.wavenumber
        energy = np.asarray(spectra.energy).reshape(-1)
        enstrophy = np.asarray(spectra.enstrophy).reshape(-1)
        if step in energy_sums[name]:
            energy_sums[name][step] += energy
            enstrophy_sums[name][step] += enstrophy
        else:
            energy_sums[name][step] = energy
            enstrophy_sums[name][step] = enstrophy

    decomposition_wavenumber: np.ndarray | None = None

    def accumulate_decomposition(
        name: str,
        step: int,
        field: np.ndarray,
        target: np.ndarray,
    ) -> None:
        # Deliberately not truncated: the corner shells above N/2 carry a real
        # share of the error (about a fifth here), and dropping them would
        # break the identity that makes this decomposition worth reporting,
        # namely that its terms sum to the endpoint field MSE at that step.
        nonlocal decomposition_wavenumber
        split_error = error_decomposition(field, target)
        if decomposition_wavenumber is None:
            decomposition_wavenumber = split_error.wavenumber
        amplitude = np.asarray(split_error.amplitude).reshape(-1)
        phase = np.asarray(split_error.phase).reshape(-1)
        if step in amplitude_sums[name]:
            amplitude_sums[name][step] += amplitude
            phase_sums[name][step] += phase
        else:
            amplitude_sums[name][step] = amplitude
            phase_sums[name][step] = phase

    for seed in requested_seeds:
        reference = generate_reference_trajectory(
            seed,
            num_steps,
            split=split,
            config=dns_config,
        )
        targets = np.asarray(reference.targets)
        for step in steps:
            accumulate(REFERENCE_METHOD, step, targets[step - 1])
        for name, rollout in methods.items():
            trajectory = np.asarray(
                jnp.asarray(rollout(reference.initial_coarse), dtype=jnp.float32)
            )
            if not np.all(np.isfinite(trajectory)):
                raise ValueError(
                    f"method {name!r} produced a non-finite trajectory at seed {seed}"
                )
            for step in steps:
                accumulate(name, step, trajectory[step - 1])
                accumulate_decomposition(
                    name, step, trajectory[step - 1], targets[step - 1]
                )

    if wavenumber is None:  # pragma: no cover - guarded by the seed check above
        raise ValueError("no spectra were accumulated")

    count = float(len(requested_seeds))
    spectra_report: dict[str, dict[str, dict[str, list[float]]]] = {}
    distances: dict[str, dict[str, dict[str, float]]] = {}
    decomposition_report: dict[str, dict[str, dict[str, object]]] = {}
    for step in steps:
        step_key = str(step)
        decomposition_report[step_key] = {}
        for name in methods:
            amplitude = amplitude_sums[name][step] / count
            phase = phase_sums[name][step] / count
            total = float(amplitude.sum() + phase.sum())
            decomposition_report[step_key][name] = {
                "amplitude": amplitude.tolist(),
                "phase": phase.tolist(),
                "amplitude_total": float(amplitude.sum()),
                "phase_total": float(phase.sum()),
                "squared_error_total": total,
                "phase_fraction": (float(phase.sum() / total) if total > 0.0 else 0.0),
            }
        reference_energy = energy_sums[REFERENCE_METHOD][step] / count
        reference_enstrophy = enstrophy_sums[REFERENCE_METHOD][step] / count
        spectra_report[step_key] = {}
        distances[step_key] = {}
        for name in names:
            energy = energy_sums[name][step] / count
            enstrophy = enstrophy_sums[name][step] / count
            spectra_report[step_key][name] = {
                "energy": energy.tolist(),
                "enstrophy": enstrophy.tolist(),
            }
            if name == REFERENCE_METHOD:
                continue
            distances[step_key][name] = {
                "energy_log_distance": spectral_distance(energy, reference_energy),
                "enstrophy_log_distance": spectral_distance(
                    enstrophy, reference_enstrophy
                ),
            }

    return {
        "diagnostic": "radial energy and enstrophy spectra",
        "split": split,
        "seeds": list(requested_seeds),
        "report_steps": list(steps),
        "wavenumber": wavenumber.tolist(),
        "max_wavenumber": max_wavenumber,
        "reference_method": REFERENCE_METHOD,
        "solver_config": vars(solver_config),
        "dns_config": vars(dns_config),
        "checkpoints": {
            "aposteriori": {
                "path": str(aposteriori_path),
                "sha256": aposteriori_digest,
            },
            "apriori": (
                None
                if apriori_checkpoint is None
                else {
                    "path": str(Path(apriori_checkpoint)),
                    "sha256": apriori_digest,
                }
            ),
        },
        "spectra": spectra_report,
        "spectral_distance": distances,
        "decomposition_wavenumber": (
            []
            if decomposition_wavenumber is None
            else decomposition_wavenumber.tolist()
        ),
        "error_decomposition": decomposition_report,
        "notes": (
            "Seed-averaged spectra and mode-wise amplitude/phase error "
            "decomposition on the validation split. At each requested step "
            "the decomposition splits that step's endpoint-field squared "
            "vorticity error (the endpoint state against the filtered-DNS "
            "target at that step), not the rollout-prefix MSE the project "
            "reports: summing its amplitude and phase terms over all shells "
            "returns the endpoint field's vorticity MSE exactly. "
            "Diagnostic only: no reported metric, model selection, or "
            "test-split result depends on it."
        ),
    }
