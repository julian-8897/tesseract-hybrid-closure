"""Reproducible final-submission evaluation protocol.

Three stages run in a fixed order, each refusing to clobber an existing
destination and each recording enough provenance for the later stages to reuse
the earlier artefacts without recomputing them:

1. ``select``   — pick the final a-posteriori checkpoint by the lowest full
   32-seed validation vorticity-MSE at the locked 30-step horizon, recording
   both candidate metrics and the criterion before any test access. Each
   validation reference is generated once per seed and shared by all
   candidates.
2. ``apriori``  — train the matched 700-update a-priori baseline on training
   seeds 0-699 with the same Adam learning rate in the calibrated regime.
3. ``evaluate`` — on the locked test seeds, generate one reference trajectory
   per seed to the maximum horizon, roll each method once per seed to that
   maximum, and reuse the prefixes for every shorter horizon.

The ``select`` and ``apriori`` stages never touch test seeds. The ``evaluate``
stage fails closed unless the selection report matches the locked protocol and
the a-priori checkpoint exactly matches the selected checkpoint, all before
any test trajectory is generated. The CLI enforces the approved locks
(horizon 30, 700 updates, horizons 30/60/120/250/500) without exposing them as
knobs; relaxed settings survive only through the internal function arguments
used by validation-only synthetic tests.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .baselines import smagorinsky_rollout
from .checkpointing import (
    read_training_checkpoint_with_digest,
    save_training_checkpoint,
    validate_sha256_digest,
)
from .closure import parameter_count
from .configs import DNSConfig, SolverConfig, seed_range_for_split
from .constants import (
    LEARNING_RATE,
    SMAGORINSKY_TEST_FILTER_RATIO,
    STATIC_SMAGORINSKY_COEFFICIENT,
)
from .data import ReferenceTrajectory, generate_reference_trajectory
from .losses import aposteriori_loss, closure_rollout, no_closure_rollout, vorticity_mse
from .solver import CoarseVorticityStepper
from .training import _resolve_matched_configs, train_apriori_baseline

SELECTION_HORIZON = 30
MATCHED_APRIORI_UPDATES = 700
FINAL_EVALUATION_HORIZONS = (30, 60, 120, 250, 500)
APRIORI_SCHEME = "apriori"

_validation_seeds = tuple(seed_range_for_split("validation"))
VALIDATION_NUM_SEEDS = len(_validation_seeds)
VALIDATION_SEED_RANGE_TEXT = f"{_validation_seeds[0]}..{_validation_seeds[-1]}"

PROTOCOL = {
    "name": "final-submission-evaluation",
    "version": 1,
    "steps": [
        f"select: lowest full-{VALIDATION_NUM_SEEDS}-seed validation "
        f"vorticity-MSE at horizon {SELECTION_HORIZON} among the a-posteriori "
        "candidates, recorded before any test access",
        "apriori: matched a-priori baseline with the same Adam learning rate, "
        f"{MATCHED_APRIORI_UPDATES} updates on consecutive training seeds "
        "from 0, calibrated regime",
        f"evaluate: locked test seeds at horizons {FINAL_EVALUATION_HORIZONS}; "
        "one reference trajectory per seed to the maximum horizon, prefixes "
        "reused; each method rolled once per seed to the maximum horizon",
    ],
    "locks": {
        "selection_horizon": SELECTION_HORIZON,
        "apriori_updates": MATCHED_APRIORI_UPDATES,
        "test_horizons": list(FINAL_EVALUATION_HORIZONS),
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sanitise_json(value):
    """Recursively replace non-finite floats with explicit JSON strings."""
    if isinstance(value, (float, np.floating)):
        scaled = float(value)
        if math.isnan(scaled):
            return "NaN"
        if math.isinf(scaled):
            return "Infinity" if scaled > 0.0 else "-Infinity"
        return scaled
    if isinstance(value, Mapping):
        return {str(key): _sanitise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise_json(item) for item in value]
    return value


def _as_json(payload: Mapping) -> str:
    return json.dumps(_sanitise_json(payload), indent=2, allow_nan=False) + "\n"


def write_report_refusing_existing(path: str | Path, payload: Mapping) -> Path:
    """Atomically write a standards-compliant JSON report, never overwriting."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(_as_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)
    return destination


def _validate_candidate_params(checkpoint: dict, checkpoint_path) -> jax.Array:
    """Validate the parameter vector and provenance of a loaded candidate."""
    params = jnp.asarray(checkpoint["params_flat"], dtype=jnp.float32)
    if params.ndim != 1 or not bool(jnp.all(jnp.isfinite(params))):
        raise ValueError(
            f"checkpoint params must be a finite 1-D vector: {checkpoint_path}"
        )
    expected = int(parameter_count())
    if params.shape != (expected,):
        raise ValueError(
            f"checkpoint params must have exactly {expected} entries, "
            f"got {params.shape[0]}: {checkpoint_path}"
        )
    if "completed_updates" not in checkpoint or "solver_config" not in checkpoint:
        raise ValueError(f"checkpoint lacks training provenance: {checkpoint_path}")
    return params


def load_candidate_params(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict, jax.Array]:
    """Load a trusted checkpoint and validate the exact parameter vector.

    When ``expected_sha256`` is given, the checkpoint file must match that
    recorded digest byte-for-byte before it is unpickled.
    """
    checkpoint, _ = read_training_checkpoint_with_digest(
        checkpoint_path, expected_sha256=expected_sha256
    )
    params = _validate_candidate_params(checkpoint, checkpoint_path)
    return checkpoint, params


def load_candidate_params_with_digest(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict, jax.Array, str]:
    """Load a candidate and the SHA-256 of the exact bytes that produced it.

    The digest is computed over the identical byte stream that was unpickled,
    so the returned parameters are genuinely bound to the digest. Passing
    ``expected_sha256`` turns the load into a byte-for-byte verification.
    """
    checkpoint, digest = read_training_checkpoint_with_digest(
        checkpoint_path, expected_sha256=expected_sha256
    )
    params = _validate_candidate_params(checkpoint, checkpoint_path)
    return checkpoint, params, digest


def _checkpoint_configs(
    checkpoint: dict,
) -> tuple[SolverConfig, DNSConfig]:
    solver_config = checkpoint.get("solver_config")
    dns_config = checkpoint.get("dns_config")
    if not solver_config or not dns_config:
        raise ValueError("checkpoint lacks persisted solver_config/dns_config")
    return SolverConfig(**solver_config), DNSConfig(**dns_config)


def _normalise_and_validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    """Validate horizons strictly: plain positive distinct Python ints."""
    resolved: list[int] = []
    for horizon in horizons:
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise ValueError(
                f"evaluation horizons must be integers, got {type(horizon).__name__}"
            )
        if horizon <= 0:
            raise ValueError("all evaluation horizons must be positive")
        resolved.append(horizon)
    if not resolved:
        raise ValueError("at least one evaluation horizon is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("evaluation horizons must be distinct")
    return tuple(sorted(resolved))


def _validate_selection_report_structure(report: Mapping) -> None:
    """Minimal structural check for relaxed validation-only evaluation."""
    selected = report.get("selected")
    candidates = report.get("candidates")
    if not isinstance(selected, Mapping) or not selected.get("name"):
        raise ValueError("selection report must carry selected.name")
    if not isinstance(candidates, Mapping) or selected["name"] not in candidates:
        raise ValueError("selection report must carry the selected candidate entry")
    candidate = candidates[selected["name"]]
    for field in ("checkpoint", "solver_config", "dns_config"):
        if field not in candidate:
            raise ValueError(f"selected candidate must carry {field}")


def _parse_mse_value(value) -> float:
    """Parse a report mean metric: numeric, or an explicit non-finite string."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        if isinstance(value, str) and value == "Infinity":
            return math.inf
        if isinstance(value, str) and value == "-Infinity":
            return -math.inf
        if isinstance(value, str) and value == "NaN":
            return math.nan
        raise ValueError(
            "mean_vorticity_mse must be numeric or an explicit non-finite "
            f"string, got {value!r}"
        )
    return float(value)


def _mse_notation(value: float) -> tuple[str, float]:
    """Classify a parsed mean so NaN/Infinity compare by kind, not identity."""
    if math.isnan(value):
        return "nan", 0.0
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf", 0.0
    return "finite", value


def validate_selection_report(
    report: Mapping,
    *,
    protocol_version: int = 1,
    horizon: int = SELECTION_HORIZON,
    num_seeds: int = VALIDATION_NUM_SEEDS,
    seed_range: str = VALIDATION_SEED_RANGE_TEXT,
) -> None:
    """Fail closed unless the report matches the locked selection protocol.

    Beyond the header fields, the protocol locks, the exact criterion text,
    the candidate metrics (numeric or explicit non-finite strings), and the
    winner consistency are all enforced, so a structurally valid but tampered
    report can never reach the test evaluation. Every candidate must also
    carry the SHA-256 digest of its checkpoint file recorded at selection
    time; a report without digests is rejected here and must only be consumed
    through :func:`validate_selection_report_legacy_structure`.
    """
    _validate_selection_report(
        report,
        protocol_version=protocol_version,
        horizon=horizon,
        num_seeds=num_seeds,
        seed_range=seed_range,
        require_digests=True,
    )


def validate_selection_report_legacy_structure(report: Mapping) -> None:
    """Validate a legacy selection report recorded without digests.

    Selection reports written before SHA-256 checkpoint binding (such as
    ``runs/final-submission/selection.json``) do not carry per-candidate
    digests. This explicitly named mode applies every locked structural
    check except digest presence and is reserved for read-only consumers
    such as tracked asset regeneration. It is never used by test evaluation,
    which always requires the recorded digest and its byte-for-byte match.
    A digest present on a candidate is still format-validated.
    """
    _validate_selection_report(report, require_digests=False)


def _validate_selection_report(
    report: Mapping,
    *,
    protocol_version: int = 1,
    horizon: int = SELECTION_HORIZON,
    num_seeds: int = VALIDATION_NUM_SEEDS,
    seed_range: str = VALIDATION_SEED_RANGE_TEXT,
    require_digests: bool,
) -> None:
    errors: list[str] = []
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("name") != PROTOCOL["name"]:
        errors.append(f"protocol name must be {PROTOCOL['name']!r}")
    if not isinstance(protocol, Mapping) or protocol.get("version") != int(
        protocol_version
    ):
        errors.append(f"protocol version must be {protocol_version}")
    if not isinstance(protocol, Mapping) or protocol.get("locks") != PROTOCOL["locks"]:
        errors.append(f"protocol locks must be {PROTOCOL['locks']}")
    if report.get("split") != "validation":
        errors.append("split must be 'validation'")
    if report.get("horizon") != int(horizon):
        errors.append(f"horizon must be {horizon}")
    if report.get("num_seeds") != int(num_seeds):
        errors.append(f"num_seeds must be {num_seeds}")
    if report.get("seed_range") != seed_range:
        errors.append(f"seed_range must be {seed_range!r}")
    locked_criterion = (
        f"lowest full-{int(num_seeds)}-seed validation vorticity-MSE at "
        f"horizon {int(horizon)}"
    )
    if report.get("criterion") != locked_criterion:
        errors.append(f"criterion must be {locked_criterion!r}")

    selected = report.get("selected")
    candidates = report.get("candidates")
    selected_name: str | None = None
    if not isinstance(selected, Mapping) or not selected.get("name"):
        errors.append("selected.name must be present")
    elif not isinstance(candidates, Mapping) or len(candidates) < 2:
        errors.append("at least two candidates are required")
    elif selected["name"] not in candidates:
        errors.append(
            f"selected candidate {selected.get('name')!r} must be in candidates"
        )
    else:
        selected_name = str(selected["name"])
        metrics: dict[str, float] = {}
        for name, candidate in candidates.items():
            if not isinstance(candidate, Mapping):
                errors.append(f"candidate {name!r} must be a mapping")
                continue
            if require_digests and "sha256" not in candidate:
                errors.append(
                    f"candidate {name!r} must carry its checkpoint sha256 digest"
                )
            elif "sha256" in candidate:
                try:
                    validate_sha256_digest(candidate["sha256"])
                except ValueError as error:
                    errors.append(f"candidate {name!r} sha256: {error}")
            try:
                metrics[name] = _parse_mse_value(candidate.get("mean_vorticity_mse"))
            except ValueError as error:
                errors.append(f"candidate {name!r}: {error}")
        try:
            expected_winner = select_final_model(metrics)
        except ValueError as error:
            errors.append(str(error))
        else:
            if selected_name != expected_winner:
                errors.append(
                    "selected candidate must be the lowest finite "
                    f"mean_vorticity_mse candidate ({expected_winner!r})"
                )
        try:
            selected_mean = _parse_mse_value(selected.get("mean_vorticity_mse"))
        except ValueError as error:
            errors.append(f"selected.mean_vorticity_mse: {error}")
        else:
            if selected_name in metrics and _mse_notation(
                selected_mean
            ) != _mse_notation(metrics[selected_name]):
                errors.append(
                    "selected.mean_vorticity_mse must equal the selected "
                    "candidate's mean_vorticity_mse"
                )
    if errors:
        raise ValueError(f"invalid selection report: {'; '.join(errors)}")


def select_final_model(metrics: Mapping[str, float]) -> str:
    """Return the candidate with the lowest finite validation MSE.

    A non-finite 32-seed mean cannot feed the locked criterion, so such a
    candidate is excluded from the comparison; the caller records this in the
    report rather than hiding it. Raises when no candidate is finite.
    """
    if len(metrics) == 0:
        raise ValueError("selection requires at least one candidate metric")
    finite: dict[str, float] = {}
    for name, value in metrics.items():
        scaled = float(value)
        if math.isfinite(scaled):
            finite[name] = scaled
    if not finite:
        raise ValueError(
            "no candidate has a finite validation MSE; the selection criterion "
            "cannot be applied"
        )
    return min(finite, key=finite.get)


def aggregate_seed_errors(
    per_seed: Mapping[str, float],
) -> dict[str, float | int | None]:
    """Aggregate per-seed errors for one method/horizon without hiding infinities.

    The all-seed mean/std are computed over every trajectory, so a diverged
    seed surfaces as a non-finite mean. ``mean_finite_vorticity_mse`` is kept
    separate and is ``None`` when no seed stayed finite.
    """
    values = np.asarray([float(value) for value in per_seed.values()], dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    with np.errstate(invalid="ignore", over="ignore"):
        mean = float(np.mean(values)) if values.size else None
        std = float(np.std(values)) if values.size else None
    return {
        "mean_vorticity_mse": mean,
        "std_vorticity_mse": std,
        "mean_finite_vorticity_mse": (
            float(np.mean(finite_values)) if finite_values.size else None
        ),
        "num_trajectories": int(values.size),
        "num_finite": int(finite_values.size),
        "num_diverged": int(values.size - finite_values.size),
    }


def compute_horizon_errors(
    trajectory: jax.Array,
    targets: jax.Array,
    horizons: tuple[int, ...],
) -> dict[int, float]:
    """Vorticity-MSE of each prefix length against the matching target prefix."""
    max_horizon = max(horizons)
    if targets.shape[0] < max_horizon:
        raise ValueError(
            f"target length {targets.shape[0]} is shorter than max horizon {max_horizon}"
        )
    if trajectory.shape[0] < max_horizon:
        raise ValueError(
            f"trajectory length {trajectory.shape[0]} is shorter than "
            f"max horizon {max_horizon}"
        )
    errors: dict[int, float] = {}
    for horizon in horizons:
        errors[horizon] = float(vorticity_mse(trajectory[:horizon], targets[:horizon]))
    return errors


def evaluate_seed_with_methods(
    methods: Mapping[str, Callable[[jax.Array], jax.Array]],
    reference: ReferenceTrajectory,
    *,
    horizons: tuple[int, ...],
) -> dict[str, dict[int, float]]:
    """Roll each method once to the max horizon and score its prefixes.

    Each callable must return a trajectory at least ``max(horizons)`` long for
    the shared initial state; the prefixes are then reused for every horizon.
    """
    errors: dict[str, dict[int, float]] = {}
    for name, rollout in methods.items():
        trajectory = rollout(reference.initial_coarse)
        trajectory = jnp.asarray(trajectory, dtype=jnp.float32)
        errors[name] = compute_horizon_errors(
            trajectory,
            reference.targets,
            horizons,
        )
    return errors


def run_model_selection(
    candidates: Mapping[str, str | Path],
    *,
    output_path: str | Path,
    horizon: int = SELECTION_HORIZON,
) -> dict:
    """Evaluate every candidate on the full validation split and lock a winner.

    Each validation reference is generated once per seed and reused for every
    candidate, so all candidates see identical targets. The horizon argument
    exists for validation-only synthetic tests; the report's criterion and
    warnings always state the actual horizon used.
    """
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite selection report: {destination}")
    if len(candidates) < 2:
        raise ValueError("selection requires at least two candidate checkpoints")
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError("selection horizon must be an integer")
    if horizon <= 0:
        raise ValueError("selection horizon must be positive")
    resolved_horizon = horizon

    resolved_configs: tuple[SolverConfig, DNSConfig] | None = None
    candidate_metadata: dict[str, dict] = {}
    candidate_params: dict[str, jax.Array] = {}
    for name, path in candidates.items():
        if not name or not name.strip():
            raise ValueError("candidate names must be non-empty")
        checkpoint, params, digest = load_candidate_params_with_digest(path)
        if int(checkpoint.get("completed_unroll", -1)) < resolved_horizon:
            raise ValueError(
                f"{name}: completed_unroll {checkpoint.get('completed_unroll')} "
                f"is below the selection horizon {resolved_horizon}"
            )
        candidate_configs = _checkpoint_configs(checkpoint)
        if resolved_configs is None:
            resolved_configs = candidate_configs
        elif candidate_configs != resolved_configs:
            raise ValueError(
                f"{name}: solver/DNS configs differ from the other candidates"
            )
        candidate_params[name] = params
        candidate_metadata[name] = {
            "checkpoint": str(Path(path)),
            "sha256": digest,
            "completed_updates": int(checkpoint["completed_updates"]),
            "completed_unroll": int(checkpoint["completed_unroll"]),
            "losses_last": float(checkpoint["losses"][-1])
            if checkpoint.get("losses")
            else None,
            "solver_config": asdict(candidate_configs[0]),
            "dns_config": asdict(candidate_configs[1]),
        }
    assert resolved_configs is not None
    solver_config, dns_config = resolved_configs
    stepper = CoarseVorticityStepper(solver_config)
    seeds = tuple(seed_range_for_split("validation"))

    per_seed_errors: dict[str, dict[str, float]] = {
        name: {} for name in candidate_params
    }
    for seed in seeds:
        reference = generate_reference_trajectory(
            seed,
            resolved_horizon,
            split="validation",
            config=dns_config,
        )
        for name in candidate_params:
            loss = aposteriori_loss(
                stepper,
                candidate_params[name],
                reference.initial_coarse,
                reference.targets,
            )
            per_seed_errors[name][str(seed)] = float(loss)

    metrics: dict[str, float] = {}
    candidate_reports: dict[str, dict] = {}
    for name in candidate_params:
        aggregate = aggregate_seed_errors(per_seed_errors[name])
        metrics[name] = float(aggregate["mean_vorticity_mse"])
        candidate_reports[name] = {
            **candidate_metadata[name],
            "mean_vorticity_mse": metrics[name],
            **aggregate,
        }

    selected = select_final_model(metrics)
    warnings: list[str] = []
    for name, value in metrics.items():
        if not math.isfinite(value):
            warnings.append(
                f"{name}: non-finite full-{len(seeds)}-seed validation MSE "
                f"({value}); excluded from the locked comparison"
            )
    if resolved_horizon != SELECTION_HORIZON:
        warnings.append(
            f"relaxed selection horizon {resolved_horizon} deviates from the "
            f"locked {SELECTION_HORIZON} (validation-only use only)"
        )
    criterion = (
        f"lowest full-{len(seeds)}-seed validation vorticity-MSE at "
        f"horizon {resolved_horizon}"
    )
    selection_report = {
        "protocol": PROTOCOL,
        "timestamp": _utc_now(),
        "split": "validation",
        "seed_range": f"{seeds[0]}..{seeds[-1]}",
        "num_seeds": len(seeds),
        "horizon": resolved_horizon,
        "criterion": criterion,
        "per_seed_errors": {name: per_seed_errors[name] for name in candidate_params},
        "candidates": candidate_reports,
        "selected": {"name": selected, "mean_vorticity_mse": metrics[selected]},
        "warnings": warnings,
    }
    write_report_refusing_existing(destination, selection_report)
    return selection_report


def write_apriori_checkpoint(
    checkpoint_path: str | Path,
    *,
    params_flat: jax.Array,
    losses: Sequence[float],
    num_updates: int,
    solver_config: SolverConfig,
    dns_config: DNSConfig,
    learning_rate: float = LEARNING_RATE,
) -> Path:
    """Persist a matched a-priori baseline with full provenance."""
    return save_training_checkpoint(
        checkpoint_path,
        {
            "params_flat": np.asarray(jax.device_get(params_flat), dtype=np.float32),
            "training_scheme": APRIORI_SCHEME,
            "solver_config": asdict(solver_config),
            "dns_config": asdict(dns_config),
            "completed_updates": int(num_updates),
            "learning_rate": float(learning_rate),
            "losses": [float(loss) for loss in losses],
            "training_seed_range": {
                "split": "train",
                "first": 0,
                "last": int(num_updates) - 1,
                "count": int(num_updates),
            },
        },
    )


def write_apriori_summary(
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    num_updates: int,
    losses: Sequence[float],
    solver_config: SolverConfig,
    dns_config: DNSConfig,
    learning_rate: float = LEARNING_RATE,
) -> Path:
    """Persist a human-readable summary alongside the a-priori checkpoint."""
    resolved_losses = [float(loss) for loss in losses]
    summary = {
        "checkpoint": str(checkpoint_path),
        "training_scheme": APRIORI_SCHEME,
        "updates": int(num_updates),
        "learning_rate": float(learning_rate),
        "solver_config": asdict(solver_config),
        "dns_config": asdict(dns_config),
        "training_seed_range": {
            "split": "train",
            "first": 0,
            "last": int(num_updates) - 1,
            "count": int(num_updates),
        },
        "first_loss": resolved_losses[0] if resolved_losses else None,
        "final_loss": resolved_losses[-1] if resolved_losses else None,
        "losses": resolved_losses,
        "timestamp": _utc_now(),
    }
    destination = Path(output_dir) / "training-summary.json"
    return write_report_refusing_existing(destination, summary)


def train_matched_apriori_baseline(
    num_updates: int,
    output_dir: str | Path,
    *,
    reference_checkpoint: str | Path | None = None,
    solver_config: SolverConfig | None = None,
    dns_config: DNSConfig | None = None,
) -> dict:
    """Train the matched a-priori baseline and persist it non-clobbering.

    The solver/DNS configuration is taken from ``reference_checkpoint`` when
    given (guaranteeing the baseline is matched by construction), otherwise
    resolved from explicit configs. The CLI always passes the locked 700
    updates; ``num_updates`` must be a plain positive int.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"refusing to reuse a-priori output directory: {destination}"
        )
    if not isinstance(num_updates, int) or isinstance(num_updates, bool):
        raise ValueError("num_updates must be an integer")
    if num_updates <= 0:
        raise ValueError("num_updates must be positive")
    if reference_checkpoint is not None and (solver_config or dns_config):
        raise ValueError(
            "reference_checkpoint and explicit configs are mutually exclusive"
        )
    if reference_checkpoint is not None:
        checkpoint, _ = load_candidate_params(reference_checkpoint)
        solver_config, dns_config = _checkpoint_configs(checkpoint)
    resolved_solver, resolved_dns = _resolve_matched_configs(
        solver_config,
        dns_config,
    )
    params, losses = train_apriori_baseline(
        num_updates,
        solver_config=resolved_solver,
        dns_config=resolved_dns,
    )
    _validate_apriori_result(params, losses, num_updates)
    checkpoint_path = write_apriori_checkpoint(
        destination / "checkpoint.pkl",
        params_flat=params,
        losses=losses,
        num_updates=num_updates,
        solver_config=resolved_solver,
        dns_config=resolved_dns,
    )
    summary_path = write_apriori_summary(
        destination,
        checkpoint_path=checkpoint_path,
        num_updates=num_updates,
        losses=losses,
        solver_config=resolved_solver,
        dns_config=resolved_dns,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
        "updates": num_updates,
        "first_loss": float(losses[0]) if losses else None,
        "final_loss": float(losses[-1]) if losses else None,
    }


def _validate_apriori_result(
    params: jax.Array,
    losses: Sequence[float],
    num_updates: int,
) -> None:
    """Validate the trained a-priori result before anything is persisted."""
    expected = int(parameter_count())
    params_array = np.asarray(jax.device_get(params), dtype=np.float32)
    if params_array.shape != (expected,):
        raise ValueError(
            f"a-priori training returned {params_array.shape[0]} parameters, "
            f"expected {expected}"
        )
    if not bool(np.all(np.isfinite(params_array))):
        raise FloatingPointError("a-priori training returned non-finite parameters")
    resolved_losses = [float(loss) for loss in losses]
    if len(resolved_losses) != int(num_updates):
        raise ValueError(
            f"a-priori training returned {len(resolved_losses)} losses, "
            f"expected {num_updates}"
        )
    if any(not math.isfinite(loss) for loss in resolved_losses):
        raise FloatingPointError("a-priori training returned non-finite losses")


def run_evaluation_stage(
    *,
    selection_report_path: str | Path,
    apriori_checkpoint_path: str | Path,
    output_path: str | Path,
    split: str = "test",
    horizons: Sequence[int] = FINAL_EVALUATION_HORIZONS,
) -> dict:
    """Evaluate the locked methods on a split at multiple reuse-prefix horizons.

    For ``split='test'`` the evaluation is sealed: the horizons must equal the
    locked FINAL_EVALUATION_HORIZONS and the selection report must pass
    validate_selection_report, all before any test trajectory is generated.
    For ``split='validation'`` the horizons may be relaxed and only the
    selection report structure is checked (an explicit sanity path).
    """
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation report: {destination}")
    resolved_horizons = _normalise_and_validate_horizons(horizons)
    max_horizon = max(resolved_horizons)
    seeds = tuple(seed_range_for_split(split))

    selection_source = Path(selection_report_path)
    if not selection_source.is_file():
        raise FileNotFoundError(f"selection report not found: {selection_source}")
    selection_report = json.loads(selection_source.read_text())

    if split == "test":
        if tuple(resolved_horizons) != FINAL_EVALUATION_HORIZONS:
            raise ValueError(
                "test evaluation is locked to horizons "
                f"{FINAL_EVALUATION_HORIZONS}, got {tuple(resolved_horizons)}"
            )
        validate_selection_report(selection_report)
    else:
        _validate_selection_report_structure(selection_report)

    selected_name = selection_report["selected"]["name"]
    selected_candidate = selection_report["candidates"][selected_name]
    solver_config = SolverConfig(**selected_candidate["solver_config"])
    dns_config = DNSConfig(**selected_candidate["dns_config"])

    selected_digest = selected_candidate.get("sha256")
    # For split='test' the strict validation above guarantees the digest is
    # present and well-formed; verification happens here, byte-for-byte,
    # before any reference or method trajectory can be generated.
    (
        _selected_checkpoint,
        selected_params,
        selected_digest_actual,
    ) = load_candidate_params_with_digest(
        selected_candidate["checkpoint"], expected_sha256=selected_digest
    )
    apriori_checkpoint, apriori_params, apriori_digest = (
        load_candidate_params_with_digest(apriori_checkpoint_path)
    )
    if apriori_checkpoint.get("training_scheme") != APRIORI_SCHEME:
        raise ValueError(
            f"apriori_checkpoint must carry training_scheme={APRIORI_SCHEME!r}, "
            f"got {apriori_checkpoint.get('training_scheme')!r}"
        )
    apriori_updates = int(apriori_checkpoint["completed_updates"])
    if apriori_updates != MATCHED_APRIORI_UPDATES:
        raise ValueError(
            f"matched a-priori baseline must carry exactly "
            f"{MATCHED_APRIORI_UPDATES} updates, got {apriori_updates}"
        )
    apriori_solver, apriori_dns = _checkpoint_configs(apriori_checkpoint)
    if apriori_solver != solver_config or apriori_dns != dns_config:
        raise ValueError(
            "a-priori checkpoint solver/DNS configs must exactly match the "
            "selected a-posteriori checkpoint"
        )

    stepper = CoarseVorticityStepper(solver_config)
    methods: dict[str, Callable[[jax.Array], jax.Array]] = {
        "aposteriori-selected": lambda state: closure_rollout(
            stepper, selected_params, state, max_horizon
        ),
        "apriori-matched": lambda state: closure_rollout(
            stepper, apriori_params, state, max_horizon
        ),
        "no-closure": lambda state: no_closure_rollout(stepper, state, max_horizon),
        "dynamic-smagorinsky": lambda state: smagorinsky_rollout(
            stepper, state, max_horizon, dynamic=True
        ),
        "static-smagorinsky": lambda state: smagorinsky_rollout(
            stepper, state, max_horizon, dynamic=False
        ),
    }

    per_seed_errors: dict[str, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        reference = generate_reference_trajectory(
            seed,
            max_horizon,
            split=split,
            config=dns_config,
        )
        seed_errors = evaluate_seed_with_methods(
            methods,
            reference,
            horizons=resolved_horizons,
        )
        per_seed_errors[str(seed)] = {
            name: {str(horizon): value for horizon, value in by_horizon.items()}
            for name, by_horizon in seed_errors.items()
        }

    method_reports: dict[str, dict] = {
        "aposteriori-selected": {
            "checkpoint": str(Path(selected_candidate["checkpoint"])),
            "sha256": selected_digest_actual,
            "completed_updates": int(selected_candidate.get("completed_updates", -1)),
        },
        "apriori-matched": {
            "checkpoint": str(Path(apriori_checkpoint_path)),
            "sha256": apriori_digest,
            "completed_updates": apriori_updates,
            "solver_config": asdict(apriori_solver),
            "dns_config": asdict(apriori_dns),
            "configs_match_selected": True,
        },
        "no-closure": {},
        "dynamic-smagorinsky": {
            "clipping": "Germano numerator clipped at zero so C_s^2 >= 0",
            "test_filter_ratio": SMAGORINSKY_TEST_FILTER_RATIO,
        },
        "static-smagorinsky": {
            "coefficient": STATIC_SMAGORINSKY_COEFFICIENT,
        },
    }
    for method in methods:
        horizons_report: dict[str, dict] = {}
        for horizon in resolved_horizons:
            by_seed: dict[str, float] = {
                str(seed): per_seed_errors[str(seed)][method][str(horizon)]
                for seed in seeds
            }
            horizons_report[str(horizon)] = aggregate_seed_errors(by_seed)
        method_reports[method]["horizons"] = horizons_report

    evaluation_report = {
        "protocol": PROTOCOL,
        "timestamp": _utc_now(),
        "split": split,
        "seed_range": f"{seeds[0]}..{seeds[-1]}",
        "num_seeds": len(seeds),
        "horizons": list(resolved_horizons),
        "max_horizon": max_horizon,
        "solver_config": asdict(solver_config),
        "dns_config": asdict(dns_config),
        "selection_validation": "strict" if split == "test" else "structure",
        "selection_evidence": selection_report,
        "selected_aposteriori_checkpoint": str(Path(selected_candidate["checkpoint"])),
        "apriori_checkpoint": str(Path(apriori_checkpoint_path)),
        "methods": method_reports,
        "per_seed_errors": per_seed_errors,
    }
    write_report_refusing_existing(destination, evaluation_report)
    return evaluation_report
