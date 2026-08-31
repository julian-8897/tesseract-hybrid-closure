"""Deterministic derivation of the curated final-submission result assets.

This module reads the locked selection and test-evaluation reports (plus the
paths recorded in them: the preserved checkpoints and training summary) and
derives the curated metrics JSON payload. It streams SHA-256 and byte sizes
from those preserved inputs into a post-evaluation integrity manifest so a
public reviewer can bind every curated number to the preserved local sources.
All functions are deterministic and unit-tested on synthetic reports; the
tracked ``scripts/generate_submission_assets.py`` orchestrates loading,
validation, figure rendering and output writing.

Owned-output invariant: this module never modifies any file inside ``runs/``.
The only writer exported here is ``write_owned_json``, which deliberately
overwrites a caller-designated generated document (the metrics JSON) and
refuses nothing else; it still writes atomically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .configs import seed_range_for_split
from .constants import LEARNING_RATE
from .final_eval import (
    FINAL_EVALUATION_HORIZONS,
    PROTOCOL,
    _mse_notation,
    _parse_mse_value,
    _sanitise_json,
    aggregate_seed_errors,
)

FINAL_METRICS_SCHEMA = "final-submission-metrics"
# Version 2 replaces the flat ``sources`` path map with ``source_evidence``
# (a post-evaluation SHA-256/byte-size integrity manifest over the preserved
# inputs) and embeds the full ``per_seed_errors`` table so every aggregate and
# paired win can be recomputed from the curated document alone.
FINAL_METRICS_VERSION = 2

SOURCE_EVIDENCE_ALGORITHM = "sha256"

SOURCE_EVIDENCE_STATEMENT = (
    "Post-evaluation integrity manifest: SHA-256 digests and byte sizes of the "
    "preserved inputs consumed by this generation, streamed from the preserved "
    "files after the sealed evaluation completed. The digests bind the curated "
    "values below to these local source files so a reviewer can recompute every "
    "aggregate and paired win; they are not proof that the checkpoints or "
    "reports were digest-locked before the test split was accessed."
)

SOURCE_ROLES = (
    "selection_report",
    "test_evaluation_report",
    "aposteriori_candidate_checkpoint",
    "selected_aposteriori_checkpoint",
    "apriori_checkpoint",
    "apriori_training_summary",
)

METHODS = (
    "aposteriori-selected",
    "apriori-matched",
    "no-closure",
    "dynamic-smagorinsky",
    "static-smagorinsky",
)
SMAGORINSKY_METHODS = ("dynamic-smagorinsky", "static-smagorinsky")

MAX_HORIZON = max(FINAL_EVALUATION_HORIZONS)

_TEST_SEEDS = tuple(seed_range_for_split("test"))
TEST_SEED_KEYS = {str(seed) for seed in _TEST_SEEDS}
TEST_SEED_RANGE_TEXT = f"{_TEST_SEEDS[0]}..{_TEST_SEEDS[-1]}"

METRIC_DEFINITION = (
    "Vorticity-MSE between a method's coarse rollout and the sharply filtered "
    "DNS reference, averaged over all states and all 64x64 spatial values in "
    "each rollout prefix (states after steps 1..h), then averaged across the "
    "32 seeds of the split. Each method is rolled once per seed to the maximum "
    "horizon and the prefixes are reused for every shorter horizon."
)


def relative_repo_path(path: str | Path, root: str | Path) -> str:
    """Return the POSIX path relative to the repository root, refusing escapes."""
    resolved_root = Path(root).resolve()
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"path is outside the repository root {resolved_root}: {resolved}"
        ) from error
    return relative.as_posix()


def _numbers_close(left, right) -> bool:
    """Compare two report numbers including explicit non-finite strings."""
    try:
        left_value = _parse_mse_value(left)
        right_value = _parse_mse_value(right)
    except ValueError:
        return False
    left_kind = _mse_notation(left_value)
    right_kind = _mse_notation(right_value)
    if left_kind != right_kind:
        return False
    if left_kind[0] != "finite":
        return True
    return math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-12)


def sha256_and_size(path: str | Path, *, chunk_size: int = 1 << 20) -> tuple[str, int]:
    """Stream SHA-256 (hex) and byte size of a file in fixed chunks.

    One read pass in ``chunk_size`` blocks, so files of any size hash with
    bounded memory.
    """
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _expected_source_role_counts() -> dict[str, int]:
    """Exact role multiplicities: two candidate records, one of every other role."""
    return {
        role: 2 if role == "aposteriori_candidate_checkpoint" else 1
        for role in SOURCE_ROLES
    }


def validate_source_evidence(source_evidence: Mapping) -> None:
    """Fail closed unless the integrity manifest has the exact expected shape.

    Every record must carry a plain relative path (no absolute path, no
    ``..`` segment, no symlink escape) and exactly one of a digest pair
    (``sha256`` + ``bytes``) or a ``ref`` to another record's key. The role
    multiplicities are exact, not merely a role set: the candidate role must
    appear exactly twice with distinct candidate names and every other role
    exactly once, so both a-posteriori candidates are bound and the selected
    checkpoint is never silently dropped. Record keys must be unique, so
    every ref resolves to exactly one record; a ref must point at an earlier
    record that itself carries a digest, and the two records must agree on
    the path, so a deduplicated path is bound to exactly one digest and no
    ref chain or forward reference can appear. A path may carry a digest
    only once.
    """
    errors: list[str] = []
    if not isinstance(source_evidence, Mapping):
        raise ValueError("source_evidence must be a mapping")
    if source_evidence.get("algorithm") != SOURCE_EVIDENCE_ALGORITHM:
        errors.append(f"algorithm must be {SOURCE_EVIDENCE_ALGORITHM!r}")
    statement = source_evidence.get("statement")
    if not isinstance(statement, str) or not statement:
        errors.append("statement must be a non-empty string")
    files = source_evidence.get("files")
    if not isinstance(files, (list, tuple)) or not files:
        errors.append("files must be a non-empty list")
    else:
        roles: Counter = Counter()
        keys: dict[str, int] = {}
        records: dict[int, dict] = {}
        digest_paths: set[str] = set()
        for index, record in enumerate(files):
            label = f"file {index}"
            if not isinstance(record, Mapping):
                errors.append(f"{label}: record must be a mapping")
                continue
            role = record.get("role")
            if not isinstance(role, str) or role not in SOURCE_ROLES:
                errors.append(f"{label}: role must be one of {sorted(SOURCE_ROLES)}")
                continue
            roles[role] += 1
            candidate = record.get("candidate")
            record_key = f"{role}:{candidate}" if isinstance(candidate, str) else role
            if record_key in keys:
                errors.append(f"{label}: duplicate record key {record_key!r}")
            keys[record_key] = index
            records[index] = record
            path = record.get("path")
            if not isinstance(path, str) or not path:
                errors.append(f"{label}: path must be a non-empty string")
            elif Path(path).is_absolute():
                errors.append(f"{label}: path must be relative, got {path!r}")
            elif ".." in Path(path).parts or not Path(path).parts:
                errors.append(
                    f"{label}: path must be confined to the repository, got {path!r}"
                )
            if role in (
                "aposteriori_candidate_checkpoint",
                "selected_aposteriori_checkpoint",
            ):
                if not isinstance(candidate, str) or not candidate:
                    errors.append(f"{label}: role {role} must carry a candidate name")
            has_digest = (
                isinstance(record.get("sha256"), str)
                and isinstance(record.get("bytes"), int)
                and not isinstance(record.get("bytes"), bool)
            )
            has_ref = isinstance(record.get("ref"), str)
            if has_digest == has_ref:
                errors.append(
                    f"{label}: record must carry exactly one of (sha256, bytes) "
                    "or a ref"
                )
                continue
            if has_digest:
                digest = record["sha256"]
                if len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    errors.append(f"{label}: sha256 must be a 64-char hex digest")
                if record["bytes"] < 0:
                    errors.append(f"{label}: bytes must be non-negative")
                if path in digest_paths:
                    errors.append(f"{label}: duplicate digest for path {path!r}")
                digest_paths.add(path)
        for index in sorted(records):
            record = records[index]
            if "ref" not in record:
                continue
            ref = record["ref"]
            target_index = keys.get(ref)
            if target_index is None:
                errors.append(f"file {index}: ref {ref!r} is not a record key")
                continue
            if target_index >= index:
                errors.append(
                    f"file {index}: ref {ref!r} must point to an earlier record"
                )
                continue
            target = records[target_index]
            if "sha256" not in target or "bytes" not in target:
                errors.append(
                    f"file {index}: ref {ref!r} must point to a digest record"
                )
            elif target.get("path") != record.get("path"):
                errors.append(
                    f"file {index}: ref {ref!r} path must match the target record path"
                )
        if dict(roles) != _expected_source_role_counts():
            errors.append(
                f"roles must appear exactly {_expected_source_role_counts()}, "
                f"got {dict(roles)}"
            )
        candidate_names = {
            records[index]["candidate"]
            for index in records
            if records[index].get("role") == "aposteriori_candidate_checkpoint"
            and "candidate" in records[index]
        }
        for index in records:
            record = records[index]
            if record.get("role") == "selected_aposteriori_checkpoint" and (
                record.get("candidate") not in candidate_names
            ):
                errors.append(
                    f"file {index}: selected candidate {record.get('candidate')!r} "
                    "is not among the candidate checkpoints"
                )
    if errors:
        raise ValueError("invalid source evidence: " + "; ".join(errors))


def build_source_evidence(
    *,
    root: str | Path,
    selection_report_path: str,
    evaluation_report_path: str,
    selection: Mapping,
    evaluation: Mapping,
) -> dict:
    """Build the post-evaluation integrity manifest over the preserved inputs.

    Streams SHA-256 and byte sizes for the selection report, the
    test-evaluation report, every a-posteriori candidate checkpoint, the
    selected a-posteriori checkpoint, the matched a-priori checkpoint and its
    training summary, all read-only. Paths in the manifest are relative to
    ``root``, resolved canonically, and internally ordered deterministically.
    When a source resolves to a path already hashed (the selected checkpoint
    is one of the candidates), the later record carries a ``ref`` to the
    first record instead of a second digest, so every path is bound to
    exactly one digest. Absolute paths, symlinked final components and paths
    resolving outside ``root`` raise; every file must exist and be regular.
    """
    resolved_root = Path(root).resolve()

    def resolve(path: str, what: str) -> tuple[Path, str]:
        if Path(path).is_absolute():
            raise ValueError(f"{what} must be a relative repository path: {path!r}")
        literal = resolved_root / path
        if literal.is_symlink():
            raise ValueError(f"{what} must not be a symlink: {path!r}")
        resolved = literal.resolve()
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"{what} resolves outside the repository root {resolved_root}: {path!r}"
            ) from error
        return resolved, relative.as_posix()

    def record_key(role: str, candidate: str | None) -> str:
        return f"{role}:{candidate}" if candidate is not None else role

    candidates = selection.get("candidates")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("selection report must carry a non-empty candidates map")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("name"), str):
        raise ValueError("selection report must carry selected.name")

    requested: list[tuple[str, str, str | None]] = []
    _, selection_relative = resolve(selection_report_path, "selection report")
    requested.append(("selection_report", selection_relative, None))
    _, evaluation_relative = resolve(evaluation_report_path, "test-evaluation report")
    requested.append(("test_evaluation_report", evaluation_relative, None))
    for name in sorted(candidates):
        checkpoint = candidates[name].get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ValueError(f"candidate {name} must carry a checkpoint path")
        _, relative = resolve(checkpoint, f"candidate {name} checkpoint")
        requested.append(("aposteriori_candidate_checkpoint", relative, name))
    selected_checkpoint = evaluation.get("selected_aposteriori_checkpoint")
    if not isinstance(selected_checkpoint, str):
        raise ValueError("evaluation report must carry selected_aposteriori_checkpoint")
    _, selected_relative = resolve(selected_checkpoint, "selected checkpoint")
    requested.append(
        ("selected_aposteriori_checkpoint", selected_relative, selected["name"])
    )
    apriori_checkpoint = evaluation.get("apriori_checkpoint")
    if not isinstance(apriori_checkpoint, str) or not apriori_checkpoint:
        raise ValueError("evaluation report must carry a non-empty apriori_checkpoint")
    _, apriori_relative = resolve(apriori_checkpoint, "a-priori checkpoint")
    requested.append(("apriori_checkpoint", apriori_relative, None))
    summary_path = str(Path(apriori_checkpoint).parent / "training-summary.json")
    _, summary_relative = resolve(summary_path, "a-priori training summary")
    requested.append(("apriori_training_summary", summary_relative, None))

    files: list[dict] = []
    by_resolved: dict[Path, str] = {}
    for role, relative, candidate in requested:
        resolved = resolved_root / relative  # canonical: no symlinks, no escapes
        record: dict = {"role": role, "path": relative}
        if candidate is not None:
            record["candidate"] = candidate
        first_key = by_resolved.get(resolved)
        if first_key is None:
            if not resolved.exists():
                raise FileNotFoundError(f"{role} not found: {relative}")
            if not resolved.is_file():
                raise ValueError(f"{role} must be a regular file: {relative}")
            digest, size = sha256_and_size(resolved)
            record["sha256"] = digest
            record["bytes"] = size
            by_resolved[resolved] = record_key(role, candidate)
        else:
            record["ref"] = first_key
        files.append(record)

    manifest = {
        "statement": SOURCE_EVIDENCE_STATEMENT,
        "algorithm": SOURCE_EVIDENCE_ALGORITHM,
        "files": files,
    }
    validate_source_evidence(manifest)
    return manifest


def validate_evaluation_report(
    report: Mapping,
    selection: Mapping | None = None,
) -> None:
    """Fail closed unless the evaluation report is internally consistent.

    Recomputes every reported aggregate from the per-seed errors and verifies
    the method set, seed coverage, horizons, protocol locks and (when given)
    the cross-report consistency with the selection report.
    """
    errors: list[str] = []

    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("name") != PROTOCOL["name"]:
        errors.append(f"protocol name must be {PROTOCOL['name']!r}")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("version") != PROTOCOL["version"]
    ):
        errors.append(f"protocol version must be {PROTOCOL['version']}")
    if not isinstance(protocol, Mapping) or protocol.get("locks") != PROTOCOL["locks"]:
        errors.append(f"protocol locks must be {PROTOCOL['locks']}")
    if report.get("split") != "test":
        errors.append("split must be 'test'")
    if report.get("seed_range") != TEST_SEED_RANGE_TEXT:
        errors.append(f"seed_range must be {TEST_SEED_RANGE_TEXT!r}")
    if report.get("num_seeds") != len(_TEST_SEEDS):
        errors.append(f"num_seeds must be {len(_TEST_SEEDS)}")
    if report.get("horizons") != list(FINAL_EVALUATION_HORIZONS):
        errors.append(f"horizons must be {list(FINAL_EVALUATION_HORIZONS)}")
    if report.get("max_horizon") != MAX_HORIZON:
        errors.append(f"max_horizon must be {MAX_HORIZON}")
    for config_name in ("solver_config", "dns_config"):
        if not isinstance(report.get(config_name), Mapping):
            errors.append(f"{config_name} must be a mapping")

    methods = report.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
        errors.append(f"methods must be exactly {sorted(METHODS)}")

    per_seed = report.get("per_seed_errors")
    if not isinstance(per_seed, Mapping) or set(per_seed) != TEST_SEED_KEYS:
        errors.append(
            f"per_seed_errors must cover exactly the {len(_TEST_SEEDS)} test seeds"
        )

    if errors:
        raise ValueError("invalid evaluation report: " + "; ".join(errors))

    horizon_keys = {str(horizon) for horizon in FINAL_EVALUATION_HORIZONS}
    parsed_per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed_key in sorted(TEST_SEED_KEYS):
        entry = per_seed.get(seed_key)
        if not isinstance(entry, Mapping):
            errors.append(f"seed {seed_key}: per-seed entry must be a mapping")
            continue
        parsed_seed: dict[str, dict[str, float]] = {}
        for method in METHODS:
            by_horizon = entry.get(method)
            if not isinstance(by_horizon, Mapping) or set(by_horizon) != horizon_keys:
                errors.append(
                    f"seed {seed_key} {method}: per-seed errors must cover "
                    f"horizons {sorted(horizon_keys)}"
                )
                continue
            parsed_horizon: dict[str, float] = {}
            for horizon_key in horizon_keys:
                try:
                    parsed_horizon[horizon_key] = _parse_mse_value(
                        by_horizon[horizon_key]
                    )
                except ValueError as error:
                    errors.append(f"seed {seed_key} {method} h={horizon_key}: {error}")
            parsed_seed[method] = parsed_horizon
        parsed_per_seed[seed_key] = parsed_seed

    if isinstance(methods, Mapping):
        for method in METHODS:
            horizons_report = methods[method].get("horizons")
            if not isinstance(horizons_report, Mapping):
                errors.append(f"method {method}: 'horizons' must be a mapping")
                continue
            for horizon in FINAL_EVALUATION_HORIZONS:
                horizon_key = str(horizon)
                stored = horizons_report.get(horizon_key)
                if not isinstance(stored, Mapping):
                    errors.append(
                        f"method {method} h={horizon}: aggregate must be a mapping"
                    )
                    continue
                by_seed = {
                    seed_key: parsed_per_seed[seed_key][method][horizon_key]
                    for seed_key in sorted(TEST_SEED_KEYS)
                }
                recomputed = aggregate_seed_errors(by_seed)
                for field in (
                    "num_trajectories",
                    "num_finite",
                    "num_diverged",
                ):
                    if stored.get(field) != recomputed[field]:
                        errors.append(
                            f"method {method} h={horizon}: {field} must be "
                            f"{recomputed[field]}, got {stored.get(field)}"
                        )
                for field in (
                    "mean_vorticity_mse",
                    "std_vorticity_mse",
                    "mean_finite_vorticity_mse",
                ):
                    if not _numbers_close(stored.get(field), recomputed[field]):
                        errors.append(
                            f"method {method} h={horizon}: {field} must be "
                            f"{recomputed[field]}, got {stored.get(field)}"
                        )

    if selection is not None:
        selected = selection.get("selected")
        candidates = selection.get("candidates")
        if not isinstance(selected, Mapping) or not isinstance(candidates, Mapping):
            errors.append("selection report must carry selected.name and candidates")
        else:
            selected_name = selected.get("name")
            candidate = candidates.get(selected_name) if selected_name else None
            if not isinstance(candidate, Mapping):
                errors.append("selection report must carry the selected candidate")
            else:
                for field in ("solver_config", "dns_config"):
                    if report.get(field) != candidate.get(field):
                        errors.append(
                            f"evaluation {field} must match the selected "
                            f"candidate's {field}"
                        )
                if report.get("selected_aposteriori_checkpoint") != candidate.get(
                    "checkpoint"
                ):
                    errors.append(
                        "evaluation selected_aposteriori_checkpoint must match "
                        "the selected candidate's checkpoint"
                    )
        if report.get("selection_validation") != "strict":
            errors.append("selection_validation must be 'strict' for the test split")
    if not isinstance(report.get("apriori_checkpoint"), str) or not report.get(
        "apriori_checkpoint"
    ):
        errors.append("apriori_checkpoint must be a non-empty string")

    if errors:
        raise ValueError("invalid evaluation report: " + "; ".join(errors))


def derive_method_metrics(report: Mapping) -> dict[str, dict[str, dict]]:
    """Recompute every method/horizon aggregate from the per-seed errors."""
    per_seed = report["per_seed_errors"]
    metrics: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        by_horizon: dict[str, dict] = {}
        for horizon in FINAL_EVALUATION_HORIZONS:
            horizon_key = str(horizon)
            by_seed = {
                seed_key: per_seed[seed_key][method][horizon_key]
                for seed_key in sorted(TEST_SEED_KEYS)
            }
            by_horizon[horizon_key] = aggregate_seed_errors(by_seed)
        metrics[method] = by_horizon
    return metrics


def relative_reduction(reference_mean: float, baseline_mean: float) -> float | None:
    """(baseline - reference) / baseline; None when either mean is non-finite."""
    reference = float(reference_mean)
    baseline = float(baseline_mean)
    if not (math.isfinite(reference) and math.isfinite(baseline) and baseline != 0.0):
        return None
    return (baseline - reference) / baseline


def derive_relative_reductions(
    metrics: Mapping[str, Mapping[str, Mapping]],
) -> dict[str, dict]:
    """A-posteriori relative MSE reductions versus the named baselines."""
    reductions: dict[str, dict] = {}
    for horizon in FINAL_EVALUATION_HORIZONS:
        horizon_key = str(horizon)
        aposteriori_mean = float(
            metrics["aposteriori-selected"][horizon_key]["mean_vorticity_mse"]
        )
        finite_smagorinsky = {
            method: float(metrics[method][horizon_key]["mean_vorticity_mse"])
            for method in SMAGORINSKY_METHODS
            if _mse_notation(metrics[method][horizon_key]["mean_vorticity_mse"])[0]
            == "finite"
        }
        best = (
            min(finite_smagorinsky, key=finite_smagorinsky.get)
            if finite_smagorinsky
            else None
        )
        reductions[horizon_key] = {
            "vs_no_closure": relative_reduction(
                aposteriori_mean,
                float(metrics["no-closure"][horizon_key]["mean_vorticity_mse"]),
            ),
            "vs_apriori_matched": relative_reduction(
                aposteriori_mean,
                float(metrics["apriori-matched"][horizon_key]["mean_vorticity_mse"]),
            ),
            "vs_best_smagorinsky": (
                relative_reduction(aposteriori_mean, finite_smagorinsky[best])
                if best is not None
                else None
            ),
            "best_smagorinsky": best,
            "best_smagorinsky_mean_vorticity_mse": (
                finite_smagorinsky[best] if best is not None else None
            ),
        }
    return reductions


def derive_paired_wins(report: Mapping) -> dict[str, dict]:
    """Per-horizon a-posteriori vs a-priori paired win counts on the seeds.

    A pair is counted only when both per-seed errors are finite; diverged
    seeds are excluded and reported in ``num_paired``.
    """
    per_seed = report["per_seed_errors"]
    wins: dict[str, dict] = {}
    for horizon in FINAL_EVALUATION_HORIZONS:
        horizon_key = str(horizon)
        aposteriori_wins = 0
        apriori_wins = 0
        ties = 0
        num_paired = 0
        for seed_key in sorted(TEST_SEED_KEYS):
            aposteriori = _parse_mse_value(
                per_seed[seed_key]["aposteriori-selected"][horizon_key]
            )
            apriori = _parse_mse_value(
                per_seed[seed_key]["apriori-matched"][horizon_key]
            )
            if not (math.isfinite(aposteriori) and math.isfinite(apriori)):
                continue
            num_paired += 1
            if aposteriori < apriori:
                aposteriori_wins += 1
            elif apriori < aposteriori:
                apriori_wins += 1
            else:
                ties += 1
        wins[horizon_key] = {
            "aposteriori_wins": aposteriori_wins,
            "apriori_wins": apriori_wins,
            "ties": ties,
            "num_paired": num_paired,
        }
    return wins


def build_final_metrics(
    selection: Mapping,
    evaluation: Mapping,
    *,
    generation: Mapping,
    source_evidence: Mapping,
) -> dict:
    """Assemble the curated metrics payload from the two validated reports.

    ``generation`` must carry ``command``, ``script`` and ``assets`` (relative
    repository paths). ``source_evidence`` is the post-evaluation integrity
    manifest produced by :func:`build_source_evidence`, validated and embedded
    verbatim. The payload embeds the full per-seed error table (deterministic
    order, sanitised values) so means, standard deviations, finite counts and
    paired wins can be recomputed from the curated document alone. Every path
    recorded in the payload is relative; any absolute source path raises.
    """
    validate_source_evidence(source_evidence)

    candidates = selection.get("candidates", {})
    for name, candidate in candidates.items():
        checkpoint = candidate.get("checkpoint")
        if not isinstance(checkpoint, str) or Path(checkpoint).is_absolute():
            raise ValueError(
                f"candidate {name} checkpoint must be a relative repository path"
            )

    report_per_seed = evaluation["per_seed_errors"]
    per_seed_errors: dict[str, dict[str, dict[str, object]]] = {}
    for seed_key in sorted(TEST_SEED_KEYS):
        seed_entry = report_per_seed[seed_key]
        per_seed_errors[seed_key] = {
            method: {
                horizon_key: _sanitise_json(seed_entry[method][horizon_key])
                for horizon_key in sorted(seed_entry[method], key=int)
            }
            for method in METHODS
        }

    derived = derive_method_metrics(evaluation)
    return {
        "schema": FINAL_METRICS_SCHEMA,
        "version": FINAL_METRICS_VERSION,
        "protocol": PROTOCOL,
        "metric_definition": METRIC_DEFINITION,
        "source_evidence": source_evidence,
        "config": {
            "learning_rate": LEARNING_RATE,
            "solver_config": evaluation["solver_config"],
            "dns_config": evaluation["dns_config"],
            "horizons": evaluation["horizons"],
            "max_horizon": evaluation["max_horizon"],
            "test_seed_range": evaluation["seed_range"],
            "num_test_seeds": evaluation["num_seeds"],
        },
        "selection": {
            "split": selection["split"],
            "seed_range": selection["seed_range"],
            "num_seeds": selection["num_seeds"],
            "horizon": selection["horizon"],
            "criterion": selection["criterion"],
            "selected": selection["selected"]["name"],
            "selected_mean_vorticity_mse": selection["selected"]["mean_vorticity_mse"],
            "candidates": {
                name: {
                    "checkpoint": candidate["checkpoint"],
                    "completed_updates": candidate["completed_updates"],
                    "completed_unroll": candidate["completed_unroll"],
                    "mean_vorticity_mse": candidate["mean_vorticity_mse"],
                    "std_vorticity_mse": candidate["std_vorticity_mse"],
                    "mean_finite_vorticity_mse": candidate["mean_finite_vorticity_mse"],
                    "num_trajectories": candidate["num_trajectories"],
                    "num_finite": candidate["num_finite"],
                    "num_diverged": candidate["num_diverged"],
                }
                for name, candidate in candidates.items()
            },
        },
        "test_results": {
            "methods": {
                method: {
                    horizon: dict(aggregate)
                    for horizon, aggregate in by_horizon.items()
                }
                for method, by_horizon in derived.items()
            }
        },
        "per_seed_errors": per_seed_errors,
        "relative_reductions": derive_relative_reductions(derived),
        "paired_aposteriori_vs_apriori": derive_paired_wins(evaluation),
        "generation": {
            "command": str(generation["command"]),
            "script": str(generation["script"]),
            "assets": sorted(str(path) for path in generation["assets"]),
        },
    }


def stage_update_offsets(
    curriculum: Sequence[int],
    updates_per_stage: Sequence[int],
) -> dict[int, tuple[int, int]]:
    """Map each curriculum stage to its inclusive 1-indexed update window.

    Windows follow stage order, never the unroll value itself: the locked
    curriculum (1, 5, 30) with 100 updates per stage maps stage 1 to updates
    1-100, stage 5 to 101-200 and stage 30 to 201-300. The continuation
    stage then starts at one past the last stage window's end.
    """
    resolved_curriculum = [int(unroll) for unroll in curriculum]
    resolved_counts = [int(count) for count in updates_per_stage]
    if len(resolved_curriculum) != len(resolved_counts):
        raise ValueError("curriculum and updates_per_stage must have equal length")
    if any(count <= 0 for count in resolved_counts):
        raise ValueError("all stage update counts must be positive")
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for unroll, count in zip(resolved_curriculum, resolved_counts, strict=True):
        offsets[unroll] = (cursor + 1, cursor + count)
        cursor += count
    return offsets


def aligned_frames(
    initial_state: object,
    post_step_states: object,
) -> np.ndarray:
    """Prepend the initial state to post-step states: frame k is state after k steps.

    Rollouts returned by the steppers contain only post-step states (length
    ``num_steps``, state after steps 1..num_steps). Prepend the identical
    initial state so frame `k` is the state after `k` steps for k in 0..num_steps,
    which is what the montage and animation index.
    """
    initial = np.asarray(initial_state, dtype=np.float32)
    post = np.asarray(post_step_states, dtype=np.float32)
    if post.ndim < 1:
        raise ValueError("post_step_states must be at least 1-D")
    if initial.shape != post.shape[1:]:
        raise ValueError(
            f"initial state shape {initial.shape} does not match the post-step "
            f"state shape {post.shape[1:]}"
        )
    return np.concatenate([initial[np.newaxis, ...], post], axis=0)


def frame_2d(frames: object, step: int) -> np.ndarray:
    """Squeeze the channel from a channel-first field frame for imshow."""
    frame = np.asarray(frames[step], dtype=np.float32)
    if frame.ndim == 3 and frame.shape[0] == 1:
        return frame[0]
    if frame.ndim != 2:
        raise ValueError(
            "expected a 2-D field or a (1, N, N) channel-first field, "
            f"got shape {frame.shape}"
        )
    return frame


def write_owned_json(path: str | Path, payload: Mapping) -> Path:
    """Atomically write a generated JSON document, deterministic and fsynced.

    Unlike the run reports this deliberately overwrites: the destination is
    one of the script's own generated outputs, never a report inside ``runs/``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = (
        json.dumps(
            _sanitise_json(payload),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)
    return destination
