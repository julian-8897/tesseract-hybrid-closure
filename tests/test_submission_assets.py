"""Unit tests for the curated submission-asset derivation.

Synthetic reports only: no test-split simulation, no training and no access to
``runs/``. The report builders mirror the schema produced by the locked
final-submission protocol.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tesseract_hybrid_closure import submission_assets as assets
from tesseract_hybrid_closure.configs import seed_range_for_split
from tesseract_hybrid_closure.final_eval import (
    FINAL_EVALUATION_HORIZONS,
    PROTOCOL,
    aggregate_seed_errors,
)

TEST_SEEDS = tuple(seed_range_for_split("test"))
HORIZON_KEYS = [str(horizon) for horizon in FINAL_EVALUATION_HORIZONS]


def _synthetic_selection_report() -> dict:
    """Minimal but schema-consistent selection report with two candidates."""
    solver_config = {
        "num_points": 64,
        "domain_extent": 6.283185307179586,
        "dt": 0.002,
        "diffusivity": 0.001,
        "order": 2,
    }
    dns_config = {
        **solver_config,
        "num_points": 256,
        "vorticity_amplitude": 20.0,
    }
    candidates = {}
    for name, mean in (("candidate-a", 0.0100), ("candidate-b", 0.0081)):
        candidates[name] = {
            "checkpoint": f"runs/synthetic/{name}.pkl",
            "completed_updates": 300 if name == "candidate-a" else 700,
            "completed_unroll": 30,
            "losses_last": 0.015,
            "solver_config": solver_config,
            "dns_config": dns_config,
            "mean_vorticity_mse": mean,
            "std_vorticity_mse": 0.002,
            "mean_finite_vorticity_mse": mean,
            "num_trajectories": 32,
            "num_finite": 32,
            "num_diverged": 0,
        }
    return {
        "protocol": PROTOCOL,
        "timestamp": "2026-08-28T16:37:19+00:00",
        "split": "validation",
        "seed_range": "10000..10031",
        "num_seeds": 32,
        "horizon": 30,
        "criterion": ("lowest full-32-seed validation vorticity-MSE at horizon 30"),
        "per_seed_errors": {},
        "candidates": candidates,
        "selected": {
            "name": "candidate-b",
            "mean_vorticity_mse": 0.0081,
        },
        "warnings": [],
    }


def _per_seed_value(method: str, horizon_key: str, seed_index: int) -> float:
    """Deterministic synthetic per-seed errors with a clean structure."""
    if method == "aposteriori-selected":
        base = {"30": 0.0070, "60": 0.0260, "120": 0.1000, "250": 0.4000, "500": 1.0000}
        return base[horizon_key] + 1.0e-3 * (seed_index % 5 - 2)
    if method == "apriori-matched":
        base = {"30": 0.0100, "60": 0.0360, "120": 0.1400, "250": 0.5800, "500": 1.5000}
        return base[horizon_key] + 1.0e-3 * (seed_index % 7)
    if method == "no-closure":
        base = {"30": 0.0800, "60": 0.3000, "120": 1.0000, "250": 2.5000, "500": 4.2000}
        return base[horizon_key] + 2.0e-3 * (seed_index % 3 - 1)
    if method == "dynamic-smagorinsky":
        base = {"30": 0.0810, "60": 0.2900, "120": 0.9500, "250": 2.3000, "500": 3.6000}
        return base[horizon_key] + 2.0e-3 * (seed_index % 3 - 1)
    if method == "static-smagorinsky":
        base = {"30": 0.0900, "60": 0.3200, "120": 0.9000, "250": 2.0000, "500": 3.0000}
        return base[horizon_key] + 2.0e-3 * (seed_index % 3 - 1)
    raise AssertionError(f"unknown method: {method}")


def synthetic_evaluation_report(
    *,
    override_seed: tuple[int, int, str, str, float] | None = None,
    drop_method: str | None = None,
    wrong_count: bool = False,
) -> dict:
    """Build a schema-consistent test evaluation report over synthetic values.

    ``override_seed`` (seed, horizon, method, value) is injected into the
    per-seed errors only, after the reported aggregates were derived, so the
    resulting report is internally inconsistent and validation must reject it
    (except where a test repairs the aggregate explicitly).
    """
    solver_config = {
        "num_points": 64,
        "domain_extent": 6.283185307179586,
        "dt": 0.002,
        "diffusivity": 0.001,
        "order": 2,
    }
    dns_config = {
        **solver_config,
        "num_points": 256,
        "vorticity_amplitude": 20.0,
    }
    clean_per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed_index, seed in enumerate(TEST_SEEDS):
        entry: dict[str, dict[str, float]] = {}
        for method in assets.METHODS:
            if method == drop_method and seed == TEST_SEEDS[0]:
                continue
            by_horizon = {}
            for horizon_key in HORIZON_KEYS:
                by_horizon[horizon_key] = _per_seed_value(
                    method, horizon_key, seed_index
                )
            entry[method] = by_horizon
        clean_per_seed[str(seed)] = entry

    methods: dict[str, dict] = {}
    for method in assets.METHODS:
        if method == drop_method:
            methods[method] = {"horizons": {}}
            continue
        horizons_report: dict[str, dict] = {}
        for horizon in FINAL_EVALUATION_HORIZONS:
            horizon_key = str(horizon)
            by_seed = {
                str(seed): clean_per_seed[str(seed)][method][horizon_key]
                for seed in TEST_SEEDS
            }
            horizons_report[horizon_key] = aggregate_seed_errors(by_seed)
        methods[method] = {"horizons": horizons_report}

    per_seed_errors = {
        seed_key: {method: dict(by_horizon) for method, by_horizon in entry.items()}
        for seed_key, entry in clean_per_seed.items()
    }
    if override_seed is not None:
        seed_key, horizon, method, value = override_seed
        per_seed_errors[str(seed_key)][method][str(horizon)] = value

    return {
        "protocol": PROTOCOL,
        "timestamp": "2026-08-28T16:49:59+00:00",
        "split": "test",
        "seed_range": f"{TEST_SEEDS[0]}..{TEST_SEEDS[-1]}",
        "num_seeds": len(TEST_SEEDS),
        "horizons": list(FINAL_EVALUATION_HORIZONS),
        "max_horizon": max(FINAL_EVALUATION_HORIZONS),
        "solver_config": solver_config,
        "dns_config": dns_config,
        "selection_validation": "strict",
        "selection_evidence": _synthetic_selection_report(),
        "selected_aposteriori_checkpoint": "runs/synthetic/candidate-b.pkl",
        "apriori_checkpoint": "runs/synthetic/apriori.pkl",
        "methods": methods,
        "per_seed_errors": per_seed_errors,
    }


def test_validate_evaluation_report_accepts_consistent_synthetic_report():
    report = synthetic_evaluation_report()
    selection = report["selection_evidence"]

    assets.validate_evaluation_report(report, selection)


def test_validate_evaluation_report_rejects_tampered_per_seed_value():
    report = synthetic_evaluation_report(
        override_seed=(TEST_SEEDS[0], 30, "aposteriori-selected", 0.5)
    )

    with pytest.raises(ValueError, match="mean_vorticity_mse"):
        assets.validate_evaluation_report(report, report["selection_evidence"])


def test_validate_evaluation_report_rejects_missing_method_and_bad_seed_count():
    report = synthetic_evaluation_report(drop_method="no-closure")
    with pytest.raises(ValueError, match="no-closure"):
        assets.validate_evaluation_report(report, report["selection_evidence"])

    report = synthetic_evaluation_report()
    del report["per_seed_errors"][str(TEST_SEEDS[-1])]
    with pytest.raises(ValueError, match="per_seed_errors"):
        assets.validate_evaluation_report(report, report["selection_evidence"])


def test_validate_evaluation_report_cross_checks_selected_checkpoint():
    report = synthetic_evaluation_report()
    report["selected_aposteriori_checkpoint"] = "runs/synthetic/wrong.pkl"

    with pytest.raises(ValueError, match="selected_aposteriori_checkpoint"):
        assets.validate_evaluation_report(report, report["selection_evidence"])

    report = synthetic_evaluation_report()
    report["selection_validation"] = "structure"
    with pytest.raises(ValueError, match="selection_validation"):
        assets.validate_evaluation_report(report, report["selection_evidence"])


def test_validate_evaluation_report_accepts_explicit_nonfinite_aggregates():
    report = synthetic_evaluation_report(
        override_seed=(TEST_SEEDS[0], 500, "no-closure", math.inf)
    )
    recomputed = aggregate_seed_errors(
        {
            str(seed): report["per_seed_errors"][str(seed)]["no-closure"]["500"]
            for seed in TEST_SEEDS
        }
    )
    report["methods"]["no-closure"]["horizons"]["500"] = recomputed

    assets.validate_evaluation_report(report, report["selection_evidence"])


def test_derive_method_metrics_recomputes_finite_counts():
    report = synthetic_evaluation_report()
    metrics = assets.derive_method_metrics(report)

    assert set(metrics) == set(assets.METHODS)
    assert set(metrics["aposteriori-selected"]) == set(HORIZON_KEYS)
    for method in assets.METHODS:
        for horizon_key in HORIZON_KEYS:
            aggregate = metrics[method][horizon_key]
            assert aggregate["num_trajectories"] == 32
            assert aggregate["num_finite"] == 32
            assert aggregate["num_diverged"] == 0
            assert math.isfinite(float(aggregate["mean_vorticity_mse"]))


def test_derive_relative_reductions_matches_hand_computation():
    report = synthetic_evaluation_report()
    metrics = assets.derive_method_metrics(report)
    reductions = assets.derive_relative_reductions(metrics)

    expected_vs_no_closure = (
        1.0
        - metrics["aposteriori-selected"]["30"]["mean_vorticity_mse"]
        / metrics["no-closure"]["30"]["mean_vorticity_mse"]
    )
    assert reductions["30"]["vs_no_closure"] == pytest.approx(expected_vs_no_closure)
    assert reductions["30"]["vs_apriori_matched"] == pytest.approx(
        1.0
        - metrics["aposteriori-selected"]["30"]["mean_vorticity_mse"]
        / metrics["apriori-matched"]["30"]["mean_vorticity_mse"]
    )
    # Static Smagorinsky is best at the long horizons in the synthetic data.
    assert reductions["120"]["best_smagorinsky"] == "static-smagorinsky"
    assert reductions["30"]["best_smagorinsky"] == "dynamic-smagorinsky"


def test_derive_relative_reductions_returns_none_for_nonfinite_baseline():
    report = synthetic_evaluation_report(
        override_seed=(TEST_SEEDS[0], 30, "no-closure", math.inf)
    )
    metrics = assets.derive_method_metrics(report)

    # The all-seed mean is non-finite once one seed diverges.
    assert metrics["no-closure"]["30"]["num_diverged"] == 1
    reductions = assets.derive_relative_reductions(metrics)
    assert reductions["30"]["vs_no_closure"] is None


def test_derive_paired_wins_counts_and_excludes_diverged_seeds():
    report = synthetic_evaluation_report()
    wins = assets.derive_paired_wins(report)

    for horizon_key in HORIZON_KEYS:
        assert wins[horizon_key]["num_paired"] == 32
        assert (
            wins[horizon_key]["aposteriori_wins"]
            + wins[horizon_key]["apriori_wins"]
            + wins[horizon_key]["ties"]
            == 32
        )
        assert wins[horizon_key]["aposteriori_wins"] == 32

    diverged = synthetic_evaluation_report(
        override_seed=(TEST_SEEDS[0], 30, "apriori-matched", math.inf)
    )
    wins = assets.derive_paired_wins(diverged)
    assert wins["30"]["num_paired"] == 31
    assert wins["30"]["aposteriori_wins"] == 31


def _generation() -> dict:
    return {
        "command": "uv run python scripts/generate_submission_assets.py",
        "script": "scripts/generate_submission_assets.py",
        "assets": ["docs/results/final-metrics.json", "docs/figures/fig.svg"],
    }


def _minimal_source_evidence() -> dict:
    """Fabricated but schema-valid source evidence for report-level tests.

    Mirrors the manifest shape produced by ``build_source_evidence``: digest
    records carry exactly one digest per path, and the selected checkpoint
    record refs the already-hashed candidate-b path.
    """
    return {
        "statement": "synthetic",
        "algorithm": "sha256",
        "files": [
            {
                "role": "selection_report",
                "path": "runs/synthetic/selection.json",
                "sha256": "ab" * 32,
                "bytes": 1,
            },
            {
                "role": "test_evaluation_report",
                "path": "runs/synthetic/test-evaluation.json",
                "sha256": "cd" * 32,
                "bytes": 1,
            },
            {
                "role": "aposteriori_candidate_checkpoint",
                "candidate": "candidate-a",
                "path": "runs/synthetic/candidate-a.pkl",
                "sha256": "11" * 32,
                "bytes": 1,
            },
            {
                "role": "aposteriori_candidate_checkpoint",
                "candidate": "candidate-b",
                "path": "runs/synthetic/candidate-b.pkl",
                "sha256": "22" * 32,
                "bytes": 1,
            },
            {
                "role": "selected_aposteriori_checkpoint",
                "candidate": "candidate-b",
                "path": "runs/synthetic/candidate-b.pkl",
                "ref": "aposteriori_candidate_checkpoint:candidate-b",
            },
            {
                "role": "apriori_checkpoint",
                "path": "runs/synthetic/apriori.pkl",
                "sha256": "33" * 32,
                "bytes": 1,
            },
            {
                "role": "apriori_training_summary",
                "path": "runs/synthetic/training-summary.json",
                "sha256": "44" * 32,
                "bytes": 1,
            },
        ],
    }


def test_build_final_metrics_records_relative_paths_and_command():
    report = synthetic_evaluation_report()
    payload = assets.build_final_metrics(
        report["selection_evidence"],
        report,
        generation=_generation(),
        source_evidence=_minimal_source_evidence(),
    )

    assert payload["schema"] == assets.FINAL_METRICS_SCHEMA
    assert payload["version"] == assets.FINAL_METRICS_VERSION > 1
    assert payload["generation"]["command"].startswith("uv run python")
    for record in payload["source_evidence"]["files"]:
        assert not Path(record["path"]).is_absolute(), (
            f"source {record['role']} must be relative"
        )
    assert payload["source_evidence"]["algorithm"] == assets.SOURCE_EVIDENCE_ALGORITHM
    assert not any(Path(path).is_absolute() for path in payload["generation"]["assets"])
    assert payload["selection"]["selected"] == "candidate-b"
    assert payload["config"]["num_test_seeds"] == 32
    assert payload["metric_definition"].startswith("Vorticity-MSE")
    for horizon_key in HORIZON_KEYS:
        assert horizon_key in payload["test_results"]["methods"]["aposteriori-selected"]
        assert horizon_key in payload["relative_reductions"]
        assert horizon_key in payload["paired_aposteriori_vs_apriori"]
    # The full per-seed table is embedded with deterministic coverage.
    embedded = payload["per_seed_errors"]
    assert [*embedded] == [str(seed) for seed in TEST_SEEDS]
    for seed_key in embedded:
        assert set(embedded[seed_key]) == set(assets.METHODS)
        for method in assets.METHODS:
            assert [*embedded[seed_key][method]] == HORIZON_KEYS


def test_build_final_metrics_rejects_absolute_checkpoint_path():
    report = synthetic_evaluation_report()
    selection = report["selection_evidence"]
    selection["candidates"]["candidate-b"]["checkpoint"] = "/tmp/absolute.pkl"

    with pytest.raises(ValueError, match="relative"):
        assets.build_final_metrics(
            selection,
            report,
            generation=_generation(),
            source_evidence=_minimal_source_evidence(),
        )


def test_build_final_metrics_rejects_invalid_source_evidence():
    report = synthetic_evaluation_report()
    with pytest.raises(ValueError, match="algorithm"):
        assets.build_final_metrics(
            report["selection_evidence"],
            report,
            generation=_generation(),
            source_evidence={
                "statement": "s",
                "algorithm": "md5",
                "files": _minimal_source_evidence()["files"],
            },
        )


def test_build_final_metrics_retains_every_per_seed_value():
    report = synthetic_evaluation_report()
    payload = assets.build_final_metrics(
        report["selection_evidence"],
        report,
        generation=_generation(),
        source_evidence=_minimal_source_evidence(),
    )
    embedded = payload["per_seed_errors"]
    for seed_key in embedded:
        for method in assets.METHODS:
            for horizon_key in HORIZON_KEYS:
                assert (
                    embedded[seed_key][method][horizon_key]
                    == report["per_seed_errors"][seed_key][method][horizon_key]
                )

    # Explicit non-finite per-seed strings survive verbatim.
    diverged = synthetic_evaluation_report(
        override_seed=(TEST_SEEDS[0], 500, "no-closure", math.inf)
    )
    payload = assets.build_final_metrics(
        diverged["selection_evidence"],
        diverged,
        generation=_generation(),
        source_evidence=_minimal_source_evidence(),
    )
    assert (
        payload["per_seed_errors"][str(TEST_SEEDS[0])]["no-closure"]["500"]
        == "Infinity"
    )


def test_curated_per_seed_recomputes_aggregates_and_paired_wins():
    from tesseract_hybrid_closure.final_eval import _parse_mse_value

    report = synthetic_evaluation_report()
    payload = assets.build_final_metrics(
        report["selection_evidence"],
        report,
        generation=_generation(),
        source_evidence=_minimal_source_evidence(),
    )
    embedded = payload["per_seed_errors"]
    for method in assets.METHODS:
        for horizon_key in HORIZON_KEYS:
            by_seed = {
                seed_key: _parse_mse_value(embedded[seed_key][method][horizon_key])
                for seed_key in sorted(embedded)
            }
            assert (
                aggregate_seed_errors(by_seed)
                == payload["test_results"]["methods"][method][horizon_key]
            )
    for horizon_key in HORIZON_KEYS:
        aposteriori_wins = apriori_wins = ties = num_paired = 0
        for seed_key in sorted(embedded):
            aposteriori = _parse_mse_value(
                embedded[seed_key]["aposteriori-selected"][horizon_key]
            )
            apriori = _parse_mse_value(
                embedded[seed_key]["apriori-matched"][horizon_key]
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
        assert payload["paired_aposteriori_vs_apriori"][horizon_key] == {
            "aposteriori_wins": aposteriori_wins,
            "apriori_wins": apriori_wins,
            "ties": ties,
            "num_paired": num_paired,
        }


def test_sha256_and_size_matches_known_bytes_and_streams_in_chunks(tmp_path: Path):
    import hashlib

    content = bytes(range(256)) * 8  # 2048 bytes, well over one small chunk
    path = tmp_path / "blob.bin"
    path.write_bytes(content)

    digest, size = assets.sha256_and_size(path, chunk_size=7)

    assert size == len(content)
    assert digest == hashlib.sha256(content).hexdigest()

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert assets.sha256_and_size(empty) == (hashlib.sha256(b"").hexdigest(), 0)


SOURCE_CONTENTS = {
    "runs/synthetic/selection.json": b'{"locked": true}',
    "runs/synthetic/test-evaluation.json": b'{"sealed": true}',
    "runs/synthetic/candidate-a.pkl": b"A" * 100,
    "runs/synthetic/candidate-b.pkl": b"B" * 200,
    "runs/synthetic/apriori.pkl": b"C" * 150,
    "runs/synthetic/training-summary.json": b'{"losses": []}',
}


def _materialise_synthetic_sources(root: Path) -> None:
    """Write every file the synthetic reports reference, under ``root``."""
    for relative, content in SOURCE_CONTENTS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _build_manifest(root: Path, selection: dict, evaluation: dict) -> dict:
    return assets.build_source_evidence(
        root=root,
        selection_report_path="runs/synthetic/selection.json",
        evaluation_report_path="runs/synthetic/test-evaluation.json",
        selection=selection,
        evaluation=evaluation,
    )


def test_build_source_evidence_hashes_preserved_inputs_deterministically(
    tmp_path: Path,
):
    import hashlib

    root = tmp_path / "repo"
    root.mkdir()
    _materialise_synthetic_sources(root)
    selection = _synthetic_selection_report()
    evaluation = synthetic_evaluation_report()

    first = _build_manifest(root, selection, evaluation)
    second = _build_manifest(root, selection, evaluation)

    assert first == second
    assert first["algorithm"] == assets.SOURCE_EVIDENCE_ALGORITHM
    assert first["statement"] == assets.SOURCE_EVIDENCE_STATEMENT
    assert [record["role"] for record in first["files"]] == [
        "selection_report",
        "test_evaluation_report",
        "aposteriori_candidate_checkpoint",
        "aposteriori_candidate_checkpoint",
        "selected_aposteriori_checkpoint",
        "apriori_checkpoint",
        "apriori_training_summary",
    ]
    assert [
        record["candidate"] for record in first["files"] if "candidate" in record
    ] == [
        "candidate-a",
        "candidate-b",
        "candidate-b",
    ]
    for record in first["files"]:
        assert not Path(record["path"]).is_absolute()
        if "sha256" in record:
            content = (root / record["path"]).read_bytes()
            assert record["sha256"] == hashlib.sha256(content).hexdigest()
            assert record["bytes"] == len(content)


def test_build_source_evidence_deduplicates_selected_checkpoint(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    _materialise_synthetic_sources(root)
    selection = _synthetic_selection_report()
    evaluation = synthetic_evaluation_report()

    manifest = _build_manifest(root, selection, evaluation)
    selected = [
        record
        for record in manifest["files"]
        if record["role"] == "selected_aposteriori_checkpoint"
    ]
    assert len(selected) == 1
    record = selected[0]
    assert record["candidate"] == "candidate-b"
    assert record["path"] == "runs/synthetic/candidate-b.pkl"
    assert "sha256" not in record and "bytes" not in record
    assert record["ref"] == "aposteriori_candidate_checkpoint:candidate-b"

    digest_paths = [
        record["path"] for record in manifest["files"] if "sha256" in record
    ]
    assert len(digest_paths) == len(set(digest_paths))
    assert digest_paths.count("runs/synthetic/candidate-b.pkl") == 1


def test_build_source_evidence_keeps_own_digest_for_distinct_selected_path(
    tmp_path: Path,
):
    import hashlib

    root = tmp_path / "repo"
    root.mkdir()
    _materialise_synthetic_sources(root)
    selection = _synthetic_selection_report()
    evaluation = synthetic_evaluation_report()
    (root / "runs/synthetic/other-selected.pkl").write_bytes(b"X" * 77)
    evaluation["selected_aposteriori_checkpoint"] = "runs/synthetic/other-selected.pkl"

    manifest = _build_manifest(root, selection, evaluation)
    record = [
        record
        for record in manifest["files"]
        if record["role"] == "selected_aposteriori_checkpoint"
    ][0]

    assert record["path"] == "runs/synthetic/other-selected.pkl"
    assert "ref" not in record
    assert record["sha256"] == hashlib.sha256(b"X" * 77).hexdigest()
    assert record["bytes"] == 77


def test_build_source_evidence_refuses_absolute_and_outside_paths(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    _materialise_synthetic_sources(root)
    selection = _synthetic_selection_report()
    evaluation = synthetic_evaluation_report()

    selection["candidates"]["candidate-a"]["checkpoint"] = str(
        tmp_path / "absolute.pkl"
    )
    with pytest.raises(ValueError, match="relative"):
        _build_manifest(root, selection, evaluation)

    selection["candidates"]["candidate-a"]["checkpoint"] = "../outside.pkl"
    with pytest.raises(ValueError, match="outside"):
        _build_manifest(root, selection, evaluation)

    # Restore the candidate path so the next case exercises the a-priori path.
    selection["candidates"]["candidate-a"]["checkpoint"] = (
        "runs/synthetic/candidate-a.pkl"
    )
    evaluation["apriori_checkpoint"] = "/tmp/absolute-summary.pkl"
    with pytest.raises(ValueError, match="relative"):
        _build_manifest(root, selection, evaluation)


def test_build_source_evidence_raises_on_missing_source(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    _materialise_synthetic_sources(root)
    (root / "runs/synthetic/candidate-a.pkl").unlink()

    with pytest.raises(FileNotFoundError):
        _build_manifest(
            root, _synthetic_selection_report(), synthetic_evaluation_report()
        )


def test_validate_source_evidence_rejects_malformed_manifests():
    payload = _minimal_source_evidence()
    assets.validate_source_evidence(payload)  # the baseline is valid

    missing = {
        "statement": payload["statement"],
        "algorithm": payload["algorithm"],
        "files": payload["files"][:-1],
    }
    with pytest.raises(ValueError, match="roles"):
        assets.validate_source_evidence(missing)

    both = _minimal_source_evidence()
    both["files"][0]["ref"] = "x"
    with pytest.raises(ValueError, match="exactly one"):
        assets.validate_source_evidence(both)

    neither = _minimal_source_evidence()
    del neither["files"][0]["sha256"]
    del neither["files"][0]["bytes"]
    with pytest.raises(ValueError, match="exactly one"):
        assets.validate_source_evidence(neither)

    absolute = _minimal_source_evidence()
    absolute["files"][0]["path"] = "/tmp/out.json"
    with pytest.raises(ValueError, match="relative"):
        assets.validate_source_evidence(absolute)

    duplicate = _minimal_source_evidence()
    duplicate["files"][1]["path"] = duplicate["files"][0]["path"]
    with pytest.raises(ValueError, match="duplicate digest"):
        assets.validate_source_evidence(duplicate)

    bad_digest = _minimal_source_evidence()
    bad_digest["files"][0]["sha256"] = "xyz"
    with pytest.raises(ValueError, match="hex"):
        assets.validate_source_evidence(bad_digest)

    bad_ref = _minimal_source_evidence()
    bad_ref["files"][4]["ref"] = "no-such-key"
    with pytest.raises(ValueError, match="not a record key"):
        assets.validate_source_evidence(bad_ref)

    wrong_algorithm = _minimal_source_evidence()
    wrong_algorithm["algorithm"] = "md5"
    with pytest.raises(ValueError, match="algorithm"):
        assets.validate_source_evidence(wrong_algorithm)


def test_relative_repo_path_refuses_paths_outside_root(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "docs").mkdir()

    assert assets.relative_repo_path(root / "docs" / "fig.svg", root) == "docs/fig.svg"
    with pytest.raises(ValueError, match="outside"):
        assets.relative_repo_path(tmp_path / "other" / "fig.svg", root)


def test_write_owned_json_is_deterministic_and_refuses_no_inputs(tmp_path: Path):
    payload = {"a": [1.5, None], "b": math.inf}
    first = assets.write_owned_json(tmp_path / "out.json", payload)
    second = assets.write_owned_json(tmp_path / "out.json", payload)

    assert json.loads(first.read_text()) == json.loads(second.read_text())
    assert '"Infinity"' in first.read_text()


def test_stage_update_offsets_uses_stage_order_not_unroll_value():
    locked = assets.stage_update_offsets((1, 5, 30), (100, 100, 100))
    assert locked == {1: (1, 100), 5: (101, 200), 30: (201, 300)}

    # Windows must depend only on stage order and counts, never the unroll value.
    renamed = assets.stage_update_offsets((7, 9, 11), (100, 100, 100))
    assert renamed == {7: (1, 100), 9: (101, 200), 11: (201, 300)}

    uneven = assets.stage_update_offsets((1, 5, 30), (40, 60, 55))
    assert uneven == {1: (1, 40), 5: (41, 100), 30: (101, 155)}

    single = assets.stage_update_offsets((30,), (400,))
    assert single == {30: (1, 400)}

    with pytest.raises(ValueError, match="equal length"):
        assets.stage_update_offsets((1, 5, 30), (100, 100))
    with pytest.raises(ValueError, match="positive"):
        assets.stage_update_offsets((1, 5), (0, 100))


def test_aligned_frames_prepends_initial_state_once():
    import numpy as np

    initial = np.full((1, 3, 3), 7.0, dtype=np.float32)
    post = np.full((5, 1, 3, 3), 2.0, dtype=np.float32)

    frames = assets.aligned_frames(initial, post)

    assert frames.shape == (6, 1, 3, 3)
    assert frames.dtype == np.float32
    assert np.array_equal(frames[0], initial)
    # Frame k equals post-step state k-1 for k >= 1.
    assert np.array_equal(frames[1:], post)


def test_aligned_frames_rejects_mismatched_initial_shape():
    import numpy as np

    with pytest.raises(ValueError, match="does not match"):
        assets.aligned_frames(
            np.zeros((1, 2, 2), dtype=np.float32),
            np.zeros((5, 1, 3, 3), dtype=np.float32),
        )


def test_frame_2d_squeezes_channel_and_rejects_wrong_shapes():
    import numpy as np

    frames = np.zeros((3, 1, 4, 4), dtype=np.float32)
    frames[2, 0] = 1.0

    frame = assets.frame_2d(frames, 2)
    assert frame.shape == (4, 4)
    assert frame[0, 0] == 1.0

    with pytest.raises(ValueError, match="2-D"):
        assets.frame_2d(np.zeros((3, 2, 4, 4), dtype=np.float32), 0)


def test_aligned_frames_match_mse_prefix_semantics():
    """frames[1:h+1] vs targets[:h] equals the report's post-step MSE prefix."""
    import numpy as np

    initial = np.zeros((1, 2, 2), dtype=np.float32)
    post = np.arange(40, dtype=np.float32).reshape(10, 1, 2, 2)
    targets = post.copy()
    frames = assets.aligned_frames(initial, post)

    for horizon in (1, 3, 10):
        expected = float(np.mean((post[:horizon] - targets[:horizon]) ** 2))
        actual = float(np.mean((frames[1 : horizon + 1] - targets[:horizon]) ** 2))
        assert actual == expected
