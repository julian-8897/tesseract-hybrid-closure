"""Multi-seed a-posteriori training driven entirely through served Tesseracts.

The optimiser demo proves the two served components compose under reverse mode
for a single accepted update. This module answers the next question a reader
will ask: can that deployment boundary carry a real optimisation, over many
updates and many trajectories, and produce a closure that improves on held-out
seeds?

Every forward evaluation and every gradient in the training loop crosses the
served boundary: for each update, the coarse solver and the PyTorch closure are
invoked over HTTP, and the parameter gradient is assembled from their VJP
endpoints by ``tesseract-jax``.

This is deliberately *not* how the submitted model was trained. The reported
model uses the equivalent in-process callback and VJP boundary, a 1 -> 5 -> 30
unroll curriculum, and 700 updates. A run here is short, uses a fixed small
unroll, and produces a weaker closure. Its purpose is to show that the
container boundary sustains genuine optimisation, not to replace that model.

Every run records evidence of what actually executed: requested outputs are
preflighted before any DNS generation, the final parameters are checkpointed
atomically to a non-overwriting path with its SHA-256 recorded in the report,
per-seed validation scores are kept alongside the aggregates, each client is
bound to its build config, served schema and (in image mode) Docker image
identity, and the source tree's git state is recorded. ``--local`` runs are
kept for testing and are labelled honestly: their evidence records mode
``local`` and no container identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .checkpointing import (
    read_training_checkpoint_with_digest,
    save_training_checkpoint,
    validate_sha256_digest,
)
from .closure import initial_parameters
from .component_loader import TESSERACTS_ROOT, read_component_config
from .configs import DNSConfig, SolverConfig
from .constants import LEARNING_RATE, VALIDATION_SEED_RANGE
from .data import generate_reference_trajectory
from .final_eval import load_candidate_params_with_digest
from .losses import vorticity_mse
from .tesseract_components import (
    REPO_ROOT,
    composed_tesseract_rollout,
)
from .tesseract_demo import (
    DEMO_DT,
    DEMO_VORTICITY_AMPLITUDE,
    tesseract_clients,
)
from .tesseract_instrumentation import (
    InstrumentedTesseract,
    composition_invariant_violations,
)
from .training import evaluate_rollout_mse

#: Unroll used for served training. Short on purpose: every step is an HTTP
#: round trip per component, so this trades curriculum depth for a run length
#: that is honest to reproduce in a demonstration.
SERVED_UNROLL = 2

#: Validation seeds the trained parameters are scored on afterwards. A subset of
#: the validation split, evaluated in process because scoring is not the thing
#: being demonstrated.
SERVED_VALIDATION_SEEDS = tuple(VALIDATION_SEED_RANGE[:8])

SERVED_VALIDATION_UNROLL = 30

#: Component names as declared by the two ``tesseract_config.yaml`` files.
SOLVER_COMPONENT = "coarse_solver"
CLOSURE_COMPONENT = "scalar_closure"

LOCAL_MODE = "local"
IMAGE_MODE = "image"

_HASH_CHUNK_BYTES = 1 << 20
_GIT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class ComponentEvidence:
    """Bind one executed client to its build config, schema and container.

    ``mode`` is ``local`` (in-process clients) or ``image`` (served
    containers). Name, version and framework come from the same
    ``tesseract_config.yaml`` the build consumed; ``config_sha256`` hashes
    that file. ``schema_sha256`` is the digest of the served OpenAPI schema
    when the client exposes one. In image mode the Docker image ID and
    RepoDigests come from the live container, so the report binds exactly
    what was served, not what a tag claims.
    """

    name: str
    version: str
    framework: str | None
    mode: str
    config_sha256: str
    schema_sha256: str | None = None
    image_reference: str | None = None
    image_id: str | None = None
    image_short_id: str | None = None
    repo_digests: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceManifest:
    """Git state of the tree the run executed from.

    When git is unavailable (no binary, no repository), ``unavailable_reason``
    states that instead of pretending the source is known.
    """

    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    git_dirty_files: tuple[str, ...] = ()
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class ServedTrainingResult:
    """Outcome of an a-posteriori run whose every gradient crossed containers."""

    updates: int
    unroll_steps: int
    training_seeds: list[int]
    learning_rate: float
    dt: float
    vorticity_amplitude: float
    use_images: bool
    loss_curve: list[float] = field(default_factory=list)
    first_loss: float = 0.0
    final_loss: float = 0.0
    mean_first_decile_loss: float = 0.0
    mean_last_decile_loss: float = 0.0
    gradient_size: int = 0
    all_gradients_finite: bool = False
    parameter_update_norm: float = 0.0
    validation_seeds: list[int] = field(default_factory=list)
    validation_unroll: int = 0
    validation_mse_before: float = 0.0
    validation_mse_after: float = 0.0
    validation_improved: bool = False
    in_process_reference_mse: float | None = None
    in_process_reference_checkpoint: str | None = None
    in_process_reference_sha256: str | None = None
    wall_clock_seconds: float = 0.0
    solver_apply_calls: int = 0
    solver_vjp_calls: int = 0
    closure_apply_calls: int = 0
    closure_vjp_calls: int = 0
    solver_vjp_min_cotangent_norm: float = 0.0
    closure_vjp_min_cotangent_norm: float = 0.0
    note: str = ""
    validation_mse_before_per_seed: list[float] = field(default_factory=list)
    validation_mse_after_per_seed: list[float] = field(default_factory=list)
    reference_mse_per_seed: list[float] | None = None
    solver_evidence: ComponentEvidence | None = None
    closure_evidence: ComponentEvidence | None = None
    source_manifest: SourceManifest = field(default_factory=SourceManifest)
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 over a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_digest(schema: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON form of a served OpenAPI schema."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _container_image_identity(
    client: object,
) -> tuple[str | None, str | None, tuple[str, ...]]:
    """Return ``(image_id, short_id, repo_digests)`` from a live container.

    ``container.image`` is preferred; when the client library cannot resolve
    the image object, the raw ``ImageID``/``Image`` container attribute is
    used instead, so Podman-style engines still bind an identifier.
    """
    container = None
    try:
        container = client.container_info()
    except Exception:
        return None, None, ()
    image = None
    try:
        image = container.image
    except Exception:
        image = None
    image_id: str | None = None
    repo_digests: tuple[str, ...] = ()
    if image is not None:
        image_id = getattr(image, "id", None)
        attrs = getattr(image, "attrs", None) or {}
        raw_digests = attrs.get("RepoDigests")
        if isinstance(raw_digests, list):
            repo_digests = tuple(str(d) for d in raw_digests if str(d))
    if not image_id:
        attrs = getattr(container, "attrs", None) or {}
        raw = attrs.get("ImageID") or attrs.get("Image")
        image_id = raw if isinstance(raw, str) and raw else None
    short_id = None
    if image_id:
        short_id = image_id[:19] if image_id.startswith("sha256:") else image_id[:12]
    return image_id, short_id, repo_digests


def collect_component_evidence(
    client: object,
    *,
    component: str,
    mode: str,
) -> ComponentEvidence:
    """Bind one client to its build config, served schema and container.

    Name, version and framework are read from the same
    ``tesseract_config.yaml`` the build consumed, and that file is hashed as
    ``config_sha256``. The served OpenAPI schema is hashed whenever the client
    exposes one. In image mode the Docker image identity comes from the live
    container; a served client that cannot be bound to an image ID or a
    RepoDigest is refused rather than recorded loosely.
    """
    if mode not in (LOCAL_MODE, IMAGE_MODE):
        raise ValueError(f"unsupported evidence mode {mode!r}")
    config = read_component_config(component)
    name = str(config["name"])
    version = str(config["version"])
    framework = config.get("metadata", {}).get("framework")
    if not isinstance(framework, str):
        framework = None
    config_sha256 = _sha256_file(TESSERACTS_ROOT / component / "tesseract_config.yaml")
    try:
        schema = client.openapi_schema
    except Exception:
        schema = None
    schema_sha256 = schema_digest(schema) if schema is not None else None
    if mode == IMAGE_MODE:
        image_id, image_short_id, repo_digests = _container_image_identity(client)
        if not image_id and not repo_digests:
            raise RuntimeError(
                f"no container image identity for {name!r}; refusing to record "
                "served evidence that cannot be bound"
            )
        return ComponentEvidence(
            name=name,
            version=version,
            framework=framework,
            mode=mode,
            image_reference=f"{name}:{version}",
            image_id=image_id,
            image_short_id=image_short_id,
            repo_digests=repo_digests,
            config_sha256=config_sha256,
            schema_sha256=schema_sha256,
        )
    return ComponentEvidence(
        name=name,
        version=version,
        framework=framework,
        mode=mode,
        config_sha256=config_sha256,
        schema_sha256=schema_sha256,
    )


def _run_git(root: Path, *arguments: str) -> str | None:
    """Run a read-only git command against the repository root."""
    try:
        proc = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_source_manifest() -> SourceManifest:
    """Record the git commit, branch and dirty state this run executes from."""
    commit = _run_git(REPO_ROOT, "rev-parse", "HEAD")
    if commit is None:
        return SourceManifest(
            unavailable_reason="git is not available for this run's source tree"
        )
    branch = _run_git(REPO_ROOT, "branch", "--show-current")
    status = _run_git(REPO_ROOT, "status", "--porcelain")
    if branch is None or status is None:
        return SourceManifest(
            unavailable_reason="git branch/status could not be read",
        )
    dirty_files = tuple(
        sorted(line[3:] for line in status.splitlines() if line.strip())
    )
    return SourceManifest(
        git_commit=commit,
        git_branch=branch or None,
        git_dirty=bool(dirty_files),
        git_dirty_files=dirty_files,
    )


def preflight_served_training_outputs(
    *,
    report_path: str | Path | None,
    checkpoint_path: str | Path | None,
) -> None:
    """Fail before any DNS generation or training if an output already exists.

    The report and the checkpoint must both be fresh paths, and they must not
    be the same file: a rerun of ``make served-training`` onto an existing
    report or checkpoint aborts here instead of silently destroying evidence
    after the run has spent its time.
    """
    if report_path is not None and checkpoint_path is not None:
        if Path(report_path).resolve() == Path(checkpoint_path).resolve():
            raise ValueError("report and checkpoint must be different files")
    for label, path in (
        ("report", report_path),
        ("checkpoint", checkpoint_path),
    ):
        if path is None:
            continue
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {label}: {destination}")


def run_served_training(
    *,
    updates: int,
    use_images: bool = True,
    unroll_steps: int = SERVED_UNROLL,
    learning_rate: float = LEARNING_RATE,
    validation_seeds: tuple[int, ...] = SERVED_VALIDATION_SEEDS,
    reference_checkpoint: str | Path | None = None,
    report_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> ServedTrainingResult:
    """Train the closure through the served components, one seed per update.

    Each update draws its own train-split trajectory, so the run optimises the
    a-posteriori objective over many flows rather than memorising one. The
    resulting parameters are then scored on held-out validation seeds against
    the same fixed initialisation the run started from.

    ``report_path`` and ``checkpoint_path`` are preflighted before any DNS
    generation: an existing report or checkpoint raises ``FileExistsError``
    and nothing expensive runs. The checkpoint is written atomically and
    never overwrites, and its SHA-256 and path are recorded in the report.
    """
    if isinstance(updates, bool) or not isinstance(updates, int):
        raise TypeError(f"updates must be an integer, got {type(updates).__name__}")
    if updates <= 0:
        raise ValueError("updates must be positive")
    if isinstance(unroll_steps, bool) or not isinstance(unroll_steps, int):
        raise TypeError("unroll_steps must be an integer")
    if unroll_steps <= 0:
        raise ValueError("unroll_steps must be positive")
    if not validation_seeds:
        raise ValueError("at least one validation seed is required")
    preflight_served_training_outputs(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )
    manifest = collect_source_manifest()

    solver_config = SolverConfig(dt=DEMO_DT)
    dns_config = DNSConfig(dt=DEMO_DT, vorticity_amplitude=DEMO_VORTICITY_AMPLITUDE)

    # One trajectory per update, generated up front so the timed loop measures
    # the served boundary rather than DNS generation.
    training_seeds = list(range(updates))
    references = [
        generate_reference_trajectory(
            seed,
            unroll_steps,
            split="train",
            config=dns_config,
        )
        for seed in training_seeds
    ]

    started = time.monotonic()
    with tesseract_clients(use_images=use_images) as (solver, closure):
        mode = IMAGE_MODE if use_images else LOCAL_MODE
        solver_evidence = collect_component_evidence(
            solver,
            component=SOLVER_COMPONENT,
            mode=mode,
        )
        closure_evidence = collect_component_evidence(
            closure,
            component=CLOSURE_COMPONENT,
            mode=mode,
        )
        solver = InstrumentedTesseract(solver)
        closure = InstrumentedTesseract(closure)

        initial_params = jnp.asarray(initial_parameters(), dtype=jnp.float32)
        params = initial_params

        def objective(
            candidate: jax.Array,
            initial_state: jax.Array,
            targets: jax.Array,
        ) -> jax.Array:
            rollout = composed_tesseract_rollout(
                solver,
                closure,
                candidate,
                initial_state,
                num_steps=unroll_steps,
                dt=DEMO_DT,
            )
            return vorticity_mse(rollout, targets)

        optimiser = optax.adam(learning_rate)
        optimiser_state = optimiser.init(params)
        loss_curve: list[float] = []
        gradient_size = 0
        all_finite = True

        for reference in references:
            loss, gradient = jax.value_and_grad(objective)(
                params,
                reference.initial_coarse,
                reference.targets,
            )
            if not bool(jnp.all(jnp.isfinite(gradient))):
                all_finite = False
                raise FloatingPointError(
                    "non-finite gradient during served training; refusing to "
                    "continue on a corrupted parameter vector"
                )
            gradient_size = int(gradient.size)
            loss_curve.append(float(loss))
            step, optimiser_state = optimiser.update(gradient, optimiser_state, params)
            params = optax.apply_updates(params, step)

        violations = composition_invariant_violations(
            solver_apply_calls=solver.apply_calls,
            closure_apply_calls=closure.apply_calls,
            solver_vjp_calls=solver.vjp_calls,
            closure_vjp_calls=closure.vjp_calls,
            solver_vjp_input_paths=solver.vjp_input_paths,
            closure_vjp_input_paths=closure.vjp_input_paths,
            solver_vjp_min_cotangent_norm=(
                min(solver.vjp_cotangent_norms) if solver.vjp_cotangent_norms else 0.0
            ),
            closure_vjp_min_cotangent_norm=(
                min(closure.vjp_cotangent_norms) if closure.vjp_cotangent_norms else 0.0
            ),
        )
        if violations:
            raise RuntimeError(
                "served training did not exercise the full composition: "
                + "; ".join(violations)
            )

        elapsed = time.monotonic() - started
        solver_apply_calls = solver.apply_calls
        solver_vjp_calls = solver.vjp_calls
        closure_apply_calls = closure.apply_calls
        closure_vjp_calls = closure.vjp_calls
        solver_min_cotangent = (
            min(solver.vjp_cotangent_norms) if solver.vjp_cotangent_norms else 0.0
        )
        closure_min_cotangent = (
            min(closure.vjp_cotangent_norms) if closure.vjp_cotangent_norms else 0.0
        )

    checkpoint_digest: str | None = None
    if checkpoint_path is not None:
        written = save_training_checkpoint(
            checkpoint_path,
            {
                "params_flat": np.asarray(jax.device_get(params)),
                "solver_config": asdict(solver_config),
                "dns_config": asdict(dns_config),
                "completed_updates": updates,
                "completed_unroll": unroll_steps,
                "training_seeds": training_seeds,
                "learning_rate": float(learning_rate),
                "use_images": use_images,
                "losses": loss_curve,
            },
        )
        _, checkpoint_digest = read_training_checkpoint_with_digest(written)

    # Held-out scoring runs in process: the boundary being demonstrated is the
    # training gradient path, and scoring through containers would only add
    # round trips to a number that is not about them.
    def per_seed_validation_mse(candidate: jax.Array) -> list[float]:
        return [
            float(
                evaluate_rollout_mse(
                    candidate,
                    split="validation",
                    seeds=(seed,),
                    unroll=SERVED_VALIDATION_UNROLL,
                    solver_config=solver_config,
                    dns_config=dns_config,
                )["mean_vorticity_mse"]
            )
            for seed in validation_seeds
        ]

    before_scores = per_seed_validation_mse(initial_params)
    after_scores = per_seed_validation_mse(params)
    before = _mean(before_scores)
    after = _mean(after_scores)
    decile = max(1, len(loss_curve) // 10)

    reference_mse: float | None = None
    reference_digest: str | None = None
    reference_scores: list[float] | None = None
    if reference_checkpoint is not None:
        _, reference_params, reference_digest = load_candidate_params_with_digest(
            reference_checkpoint
        )
        reference_scores = per_seed_validation_mse(reference_params)
        reference_mse = _mean(reference_scores)

    note_seed = (
        "Every training gradient crossed the served Tesseract boundary "
        "(container images)."
        if use_images
        else "Training ran against in-process clients: no containers were "
        "served for this run and mode is recorded as 'local'."
    )
    note = (
        note_seed + " This is a demonstration of that boundary, not the submitted "
        "model: the reported closure uses the equivalent in-process "
        "boundary, a 1 -> 5 -> 30 unroll curriculum, and 700 updates."
    )

    result = ServedTrainingResult(
        updates=updates,
        unroll_steps=unroll_steps,
        training_seeds=training_seeds,
        learning_rate=float(learning_rate),
        dt=DEMO_DT,
        vorticity_amplitude=DEMO_VORTICITY_AMPLITUDE,
        use_images=use_images,
        loss_curve=loss_curve,
        first_loss=loss_curve[0],
        final_loss=loss_curve[-1],
        mean_first_decile_loss=_mean(loss_curve[:decile]),
        mean_last_decile_loss=_mean(loss_curve[-decile:]),
        gradient_size=gradient_size,
        all_gradients_finite=all_finite,
        parameter_update_norm=float(jnp.linalg.norm(params - initial_params)),
        validation_seeds=list(validation_seeds),
        validation_unroll=SERVED_VALIDATION_UNROLL,
        validation_mse_before=before,
        validation_mse_after=after,
        validation_improved=after < before,
        in_process_reference_mse=reference_mse,
        in_process_reference_checkpoint=(
            None if reference_checkpoint is None else str(reference_checkpoint)
        ),
        in_process_reference_sha256=reference_digest,
        wall_clock_seconds=elapsed,
        solver_apply_calls=solver_apply_calls,
        solver_vjp_calls=solver_vjp_calls,
        closure_apply_calls=closure_apply_calls,
        closure_vjp_calls=closure_vjp_calls,
        solver_vjp_min_cotangent_norm=solver_min_cotangent,
        closure_vjp_min_cotangent_norm=closure_min_cotangent,
        note=note,
        validation_mse_before_per_seed=before_scores,
        validation_mse_after_per_seed=after_scores,
        reference_mse_per_seed=reference_scores,
        solver_evidence=solver_evidence,
        closure_evidence=closure_evidence,
        source_manifest=manifest,
        checkpoint_path=(None if checkpoint_path is None else str(checkpoint_path)),
        checkpoint_sha256=checkpoint_digest,
    )
    validate_served_training_evidence(result)
    if report_path is not None:
        write_served_training_report(result, report_path)
    return result


def write_served_training_report(
    result: ServedTrainingResult, path: str | Path
) -> Path:
    """Write validated evidence as JSON, refusing to overwrite a report."""
    validate_served_training_evidence(result)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n"
    )
    return destination


def _validate_component_evidence(label: str, evidence: ComponentEvidence) -> None:
    if evidence.mode not in (LOCAL_MODE, IMAGE_MODE):
        raise ValueError(f"{label} evidence has unknown mode {evidence.mode!r}")
    if evidence.mode == IMAGE_MODE:
        if not evidence.image_id and not evidence.repo_digests:
            raise ValueError(
                f"{label} image evidence must bind an image ID or a RepoDigest"
            )
        if evidence.image_reference is None:
            raise ValueError(f"{label} image evidence must record its image reference")
    else:
        if evidence.image_id or evidence.repo_digests:
            raise ValueError(
                f"{label} local evidence must not claim container identity"
            )
    for field_name in ("config_sha256", "schema_sha256"):
        digest = getattr(evidence, field_name)
        if digest is not None:
            validate_sha256_digest(digest)


def validate_served_training_evidence(result: ServedTrainingResult) -> None:
    """Validate every evidence field, raising on any inconsistency.

    Catches tampered digests, per-seed lists that do not match the declared
    seeds or aggregates, checkpoints whose recorded path or SHA-256 no longer
    matches the file on disk, and source manifests that contradict their dirty
    state. Call this on any report whose numbers will be quoted.
    """
    expected = len(result.validation_seeds)
    if expected == 0:
        raise ValueError("validation seeds must be non-empty for evidence validation")
    for label, scores in (
        ("validation_mse_before_per_seed", result.validation_mse_before_per_seed),
        ("validation_mse_after_per_seed", result.validation_mse_after_per_seed),
    ):
        if len(scores) != expected:
            raise ValueError(f"{label} must have one entry per validation seed")
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"{label} contains non-finite values")
    if not math.isclose(
        result.validation_mse_before,
        _mean(result.validation_mse_before_per_seed),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "validation_mse_before must equal the mean of its per-seed values"
        )
    if not math.isclose(
        result.validation_mse_after,
        _mean(result.validation_mse_after_per_seed),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "validation_mse_after must equal the mean of its per-seed values"
        )

    if result.in_process_reference_sha256 is not None:
        validate_sha256_digest(result.in_process_reference_sha256)
    if result.in_process_reference_checkpoint is not None:
        if (
            result.reference_mse_per_seed is None
            or len(result.reference_mse_per_seed) != expected
        ):
            raise ValueError("reference_mse_per_seed must have one entry per seed")
        if not all(math.isfinite(score) for score in result.reference_mse_per_seed):
            raise ValueError("reference_mse_per_seed contains non-finite values")
        if result.in_process_reference_mse is None or not math.isclose(
            result.in_process_reference_mse,
            _mean(result.reference_mse_per_seed),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "in_process_reference_mse must equal the mean of its per-seed values"
            )
        if result.in_process_reference_sha256 is None:
            raise ValueError("a reference checkpoint must record its SHA-256")
    elif any(
        value is not None
        for value in (
            result.reference_mse_per_seed,
            result.in_process_reference_mse,
            result.in_process_reference_sha256,
        )
    ):
        raise ValueError("reference evidence requires a reference checkpoint path")
    if result.checkpoint_sha256 is not None:
        validate_sha256_digest(result.checkpoint_sha256)
        if result.checkpoint_path is None:
            raise ValueError("checkpoint_sha256 requires a recorded checkpoint_path")
        source = Path(result.checkpoint_path)
        if not source.is_file():
            raise ValueError(f"checkpoint not found at recorded path: {source}")
        _, actual = read_training_checkpoint_with_digest(source)
        if actual != result.checkpoint_sha256:
            raise ValueError(
                "checkpoint SHA-256 does not match the recorded file bytes"
            )
    if result.checkpoint_path is not None and result.checkpoint_sha256 is None:
        raise ValueError("a persisted checkpoint must record its SHA-256")

    expected_mode = IMAGE_MODE if result.use_images else LOCAL_MODE
    for label, expected_name, evidence in (
        ("solver", SOLVER_COMPONENT, result.solver_evidence),
        ("closure", CLOSURE_COMPONENT, result.closure_evidence),
    ):
        if evidence is None:
            raise ValueError(f"{label} evidence is missing")
        _validate_component_evidence(label, evidence)
        if evidence.mode != expected_mode:
            raise ValueError(
                f"{label} evidence mode {evidence.mode!r} contradicts "
                f"use_images={result.use_images}"
            )
        if evidence.name != expected_name:
            raise ValueError(
                f"{label} evidence names {evidence.name!r}, expected {expected_name!r}"
            )

    manifest = result.source_manifest
    if manifest.unavailable_reason is not None:
        if manifest.git_commit is not None or manifest.git_dirty is not None:
            raise ValueError(
                "an unavailable source manifest must not carry partial git fields"
            )
    else:
        if manifest.git_commit is None or manifest.git_dirty is None:
            raise ValueError("source manifest must record commit and dirty state")
        if re.fullmatch(r"[0-9a-f]{40}", manifest.git_commit) is None:
            raise ValueError(
                f"git commit must be 40 hex characters, got {manifest.git_commit!r}"
            )
        if manifest.git_dirty and not manifest.git_dirty_files:
            raise ValueError("git_dirty requires the list of dirty files")
        if not manifest.git_dirty and manifest.git_dirty_files:
            raise ValueError("git_dirty_files must be empty when the tree is clean")
