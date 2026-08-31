"""Unit tests for the final-submission evaluation protocol.

All tests use synthetic arrays or validation/train seeds only; no test-split
data is ever accessed, and no training or full evaluation is executed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tesseract_hybrid_closure import cli as cli_module
from tesseract_hybrid_closure import final_eval as final_eval_module
from tesseract_hybrid_closure.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from tesseract_hybrid_closure.cli import _candidates_from_args, build_parser, main
from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
from tesseract_hybrid_closure.constants import (
    LEARNING_RATE,
    SMAGORINSKY_TEST_FILTER_RATIO,
    STATIC_SMAGORINSKY_COEFFICIENT,
)
from tesseract_hybrid_closure.data import ReferenceTrajectory
from tesseract_hybrid_closure.final_eval import (
    APRIORI_SCHEME,
    FINAL_EVALUATION_HORIZONS,
    MATCHED_APRIORI_UPDATES,
    PROTOCOL,
    SELECTION_HORIZON,
    _checkpoint_configs,
    _normalise_and_validate_horizons,
    _sanitise_json,
    aggregate_seed_errors,
    compute_horizon_errors,
    evaluate_seed_with_methods,
    load_candidate_params,
    load_candidate_params_with_digest,
    run_evaluation_stage,
    run_model_selection,
    select_final_model,
    train_matched_apriori_baseline,
    validate_selection_report,
    validate_selection_report_legacy_structure,
    write_apriori_checkpoint,
    write_apriori_summary,
    write_report_refusing_existing,
)


def test_protocol_locks_are_pinned():
    assert SELECTION_HORIZON == 30
    assert MATCHED_APRIORI_UPDATES == 700
    assert FINAL_EVALUATION_HORIZONS == (30, 60, 120, 250, 500)


def test_select_final_model_picks_lowest_finite_metric():
    assert select_final_model({"a": 1.5, "b": 2.0}) == "a"
    assert select_final_model({"a": 3.0, "b": 2.0, "c": 1.0}) == "c"


def test_select_final_model_excludes_nonfinite_candidates():
    assert select_final_model({"a": 1.5, "b": math.inf}) == "a"
    assert select_final_model({"a": math.nan, "b": 2.0}) == "b"


def test_select_final_model_rejects_empty_and_all_nonfinite():
    with pytest.raises(ValueError, match="at least one"):
        select_final_model({})
    with pytest.raises(ValueError, match="finite"):
        select_final_model({"a": math.nan, "b": math.inf})


def test_aggregate_seed_errors_handles_diverged_seeds():
    aggregate = aggregate_seed_errors({"1": 1.0, "2": 2.0, "3": math.inf})

    assert aggregate["num_trajectories"] == 3
    assert aggregate["num_finite"] == 2
    assert aggregate["num_diverged"] == 1
    assert math.isinf(float(aggregate["mean_vorticity_mse"]))
    assert math.isnan(float(aggregate["std_vorticity_mse"]))
    assert aggregate["mean_finite_vorticity_mse"] == pytest.approx(1.5)


def test_aggregate_seed_errors_all_diverged_and_empty():
    all_diverged = aggregate_seed_errors({"1": math.inf, "2": math.nan})
    assert all_diverged["num_diverged"] == 2
    assert all_diverged["num_finite"] == 0
    assert all_diverged["mean_finite_vorticity_mse"] is None

    empty = aggregate_seed_errors({})
    assert empty["num_trajectories"] == 0
    assert empty["mean_vorticity_mse"] is None
    assert empty["std_vorticity_mse"] is None


def test_compute_horizon_errors_prefix_semantics():
    targets = jnp.zeros((5, 1, 2, 2), dtype=jnp.float32)
    trajectory = jnp.zeros((5, 1, 2, 2), dtype=jnp.float32)
    trajectory = trajectory.at[:2].set(1.0)

    errors = compute_horizon_errors(trajectory, targets, (1, 3, 5))

    assert set(errors) == {1, 3, 5}
    assert errors[1] == pytest.approx(1.0)
    assert errors[3] == pytest.approx(8 / 12)
    assert errors[5] == pytest.approx(0.4)


def test_compute_horizon_errors_rejects_short_arrays():
    targets = jnp.zeros((5, 1, 2, 2), dtype=jnp.float32)
    short = jnp.zeros((3, 1, 2, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="shorter than"):
        compute_horizon_errors(short, targets, (5,))
    with pytest.raises(ValueError, match="shorter than"):
        compute_horizon_errors(targets, short, (5,))


def test_normalise_and_validate_horizons():
    assert _normalise_and_validate_horizons((120, 30, 60)) == (30, 60, 120)
    with pytest.raises(ValueError, match="required"):
        _normalise_and_validate_horizons(())
    with pytest.raises(ValueError, match="positive"):
        _normalise_and_validate_horizons((0, 30))
    with pytest.raises(ValueError, match="distinct"):
        _normalise_and_validate_horizons((30, 30))


def test_evaluate_seed_with_methods_rolls_each_method_once():
    reference = ReferenceTrajectory(
        initial_coarse=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        targets=jnp.zeros((5, 1, 2, 2), dtype=jnp.float32),
    )
    calls = {"ones": 0, "zeros": 0}

    def ones_rollout(_state):
        calls["ones"] += 1
        trajectory = jnp.zeros((5, 1, 2, 2), dtype=jnp.float32)
        return trajectory.at[:2].set(1.0)

    def zeros_rollout(_state):
        calls["zeros"] += 1
        return jnp.zeros((5, 1, 2, 2), dtype=jnp.float32)

    errors = evaluate_seed_with_methods(
        {"ones": ones_rollout, "zeros": zeros_rollout},
        reference,
        horizons=(1, 3, 5),
    )

    assert calls == {"ones": 1, "zeros": 1}
    assert errors["ones"][1] == pytest.approx(1.0)
    assert errors["ones"][3] == pytest.approx(8 / 12)
    assert errors["ones"][5] == pytest.approx(0.4)
    assert errors["zeros"][1] == pytest.approx(0.0)
    assert errors["zeros"][5] == pytest.approx(0.0)


def test_evaluate_seed_with_methods_rejects_short_trajectories():
    reference = ReferenceTrajectory(
        initial_coarse=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        targets=jnp.zeros((5, 1, 2, 2), dtype=jnp.float32),
    )

    with pytest.raises(ValueError, match="shorter than"):
        evaluate_seed_with_methods(
            {"short": lambda _state: jnp.zeros((3, 1, 2, 2), dtype=jnp.float32)},
            reference,
            horizons=(5,),
        )


def test_write_report_refusing_existing_is_atomic_and_guarded(tmp_path: Path):
    destination = tmp_path / "report.json"
    payload = {"protocol": {"name": "final-test"}, "values": [1.0, 2.0]}

    written = write_report_refusing_existing(destination, payload)

    assert written == destination
    assert json.loads(destination.read_text()) == payload
    assert list(tmp_path.iterdir()) == [destination]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_report_refusing_existing(destination, {"other": True})


def test_load_candidate_params_validates_finite_1d_params(tmp_path: Path):
    path = save_training_checkpoint(
        tmp_path / "params2d.pkl",
        {
            "params_flat": jnp.ones((2, 2), dtype=jnp.float32),
            "completed_updates": 1,
            "solver_config": asdict(SolverConfig(dt=0.002)),
        },
    )

    with pytest.raises(ValueError, match="finite 1-D"):
        load_candidate_params(path)


def test_checkpoint_config_provenance_round_trip(tmp_path: Path):
    solver = SolverConfig(dt=0.002)
    dns = DNSConfig(dt=0.002, vorticity_amplitude=20.0)
    path = save_training_checkpoint(
        tmp_path / "provenance.pkl",
        {
            "params_flat": jnp.arange(3, dtype=jnp.float32),
            "completed_updates": 700,
            "solver_config": asdict(solver),
            "dns_config": asdict(dns),
        },
    )

    loaded_solver, loaded_dns = _checkpoint_configs(load_training_checkpoint(path))

    assert loaded_solver == solver
    assert loaded_dns == dns


def test_write_apriori_checkpoint_persists_provenance(tmp_path: Path):
    params = jnp.arange(7, dtype=jnp.float32)
    solver = SolverConfig(dt=0.002)
    dns = DNSConfig(dt=0.002, vorticity_amplitude=20.0)
    path = tmp_path / "apriori" / "checkpoint.pkl"

    checkpoint_path = write_apriori_checkpoint(
        path,
        params_flat=params,
        losses=[1.25, 2.5],
        num_updates=700,
        solver_config=solver,
        dns_config=dns,
    )

    assert checkpoint_path == path
    checkpoint = load_training_checkpoint(path)
    assert checkpoint["training_scheme"] == APRIORI_SCHEME
    assert checkpoint["completed_updates"] == 700
    assert checkpoint["learning_rate"] == LEARNING_RATE
    assert checkpoint["losses"] == [1.25, 2.5]
    assert checkpoint["training_seed_range"] == {
        "split": "train",
        "first": 0,
        "last": 699,
        "count": 700,
    }
    assert jnp.array_equal(checkpoint["params_flat"], params)
    assert _checkpoint_configs(checkpoint) == (solver, dns)
    with pytest.raises(FileExistsError):
        write_apriori_checkpoint(
            path,
            params_flat=params,
            losses=[1.25, 2.5],
            num_updates=700,
            solver_config=solver,
            dns_config=dns,
        )


def test_write_apriori_summary_persists_provenance(tmp_path: Path):
    solver = SolverConfig(dt=0.002)
    dns = DNSConfig(dt=0.002, vorticity_amplitude=20.0)
    output_dir = tmp_path / "apriori"

    summary_path = write_apriori_summary(
        output_dir,
        checkpoint_path="runs/final/checkpoint.pkl",
        num_updates=700,
        losses=[1.0, 2.0],
        solver_config=solver,
        dns_config=dns,
    )

    summary = json.loads(Path(summary_path).read_text())
    assert summary["training_scheme"] == APRIORI_SCHEME
    assert summary["updates"] == 700
    assert summary["learning_rate"] == LEARNING_RATE
    assert summary["first_loss"] == 1.0
    assert summary["final_loss"] == 2.0
    assert summary["solver_config"]["dt"] == 0.002
    assert summary["dns_config"]["vorticity_amplitude"] == 20.0
    assert summary["training_seed_range"]["last"] == 699
    with pytest.raises(FileExistsError):
        write_apriori_summary(
            output_dir,
            checkpoint_path="runs/final/checkpoint.pkl",
            num_updates=700,
            losses=[1.0, 2.0],
            solver_config=solver,
            dns_config=dns,
        )


def test_train_matched_apriori_refuses_existing_output_dir(tmp_path: Path):
    output_dir = tmp_path / "apriori-out"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="a-priori output directory"):
        train_matched_apriori_baseline(700, output_dir)


def test_train_matched_apriori_rejects_bad_updates(tmp_path: Path):
    with pytest.raises(ValueError, match="must be positive"):
        train_matched_apriori_baseline(0, tmp_path / "never")


def test_train_matched_apriori_rejects_conflicting_config_sources(tmp_path: Path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_matched_apriori_baseline(
            700,
            tmp_path / "never",
            reference_checkpoint="some.pkl",
            solver_config=SolverConfig(dt=0.002),
        )


def test_run_model_selection_refuses_existing_output_before_loading(
    tmp_path: Path,
):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")

    with pytest.raises(FileExistsError, match="selection report"):
        run_model_selection(
            {"a": "missing-a.pkl", "b": "missing-b.pkl"},
            output_path=selection_path,
        )


def test_run_model_selection_requires_two_candidates(tmp_path: Path):
    with pytest.raises(ValueError, match="at least two"):
        run_model_selection({"a": "missing-a.pkl"}, output_path=tmp_path / "s.json")


def test_run_evaluation_stage_refuses_existing_output_before_reading(
    tmp_path: Path,
):
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text("{}")

    with pytest.raises(FileExistsError, match="evaluation report"):
        run_evaluation_stage(
            selection_report_path=tmp_path / "missing-selection.json",
            apriori_checkpoint_path=tmp_path / "missing-apriori.pkl",
            output_path=evaluation_path,
            split="validation",
            horizons=(30,),
        )


def test_run_evaluation_stage_requires_existing_selection_report(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="selection report"):
        run_evaluation_stage(
            selection_report_path=tmp_path / "missing-selection.json",
            apriori_checkpoint_path=tmp_path / "missing-apriori.pkl",
            output_path=tmp_path / "evaluation.json",
            split="validation",
            horizons=(30,),
        )


def test_run_evaluation_stage_rejects_apriori_scheme_mismatch(
    tmp_path: Path,
    tiny_parameter_count,
):
    selected_path = _candidate_checkpoint(
        tmp_path / "selected.pkl", jnp.arange(4, dtype=jnp.float32), 700
    )
    selection = _synthetic_selection_report(tmp_path, selected_path)
    apriori_path = save_training_checkpoint(
        tmp_path / "wrong-scheme.pkl",
        {
            "params_flat": jnp.arange(4, dtype=jnp.float32),
            "completed_updates": 700,
            "training_scheme": "aposteriori",
            "solver_config": asdict(SolverConfig(dt=0.002)),
            "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=20.0)),
        },
    )
    with pytest.raises(ValueError, match="training_scheme"):
        run_evaluation_stage(
            selection_report_path=selection,
            apriori_checkpoint_path=apriori_path,
            output_path=tmp_path / "evaluation.json",
            split="validation",
            horizons=(30,),
        )


def _synthetic_selection_report(
    tmp_path: Path,
    selected_checkpoint: str | Path = "missing-checkpoint.pkl",
    **overrides,
) -> Path:
    selected_source = Path(selected_checkpoint)
    selected_digest = (
        hashlib.sha256(selected_source.read_bytes()).hexdigest()
        if selected_source.is_file()
        else "0" * 64
    )
    report = {
        "protocol": PROTOCOL,
        "split": "validation",
        "seed_range": "10000..10031",
        "num_seeds": 32,
        "horizon": 30,
        "criterion": "lowest full-32-seed validation vorticity-MSE at horizon 30",
        "selected": {"name": "winner", "mean_vorticity_mse": 1.0},
        "candidates": {
            "winner": {
                "checkpoint": str(Path(selected_checkpoint)),
                "sha256": selected_digest,
                "solver_config": asdict(SolverConfig(dt=0.002)),
                "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=20.0)),
                "mean_vorticity_mse": 1.0,
            },
            "loser": {
                "checkpoint": str(Path(selected_checkpoint)),
                "sha256": selected_digest,
                "solver_config": asdict(SolverConfig(dt=0.002)),
                "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=20.0)),
                "mean_vorticity_mse": 2.0,
            },
        },
    }
    report.update(overrides)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(report))
    return selection


def test_candidates_from_args_parsing_and_guards():
    assert _candidates_from_args(["a:p1", "b:p2"]) == {"a": "p1", "b": "p2"}

    with pytest.raises(ValueError, match="NAME:PATH"):
        _candidates_from_args(["broken"])
    with pytest.raises(ValueError, match="NAME:PATH"):
        _candidates_from_args([":missing-name"])
    with pytest.raises(ValueError, match="duplicate"):
        _candidates_from_args(["a:p1", "a:p2"])


def test_final_parser_stage_defaults_and_subsets():
    args = build_parser().parse_args(["final", "--selection-output", "s.json"])
    assert args.stage == ["select", "apriori", "evaluate"]

    subset = build_parser().parse_args(
        ["final", "--stage", "select", "evaluate", "--selection-output", "s.json"]
    )
    assert subset.stage == ["select", "evaluate"]


def test_final_select_requires_selection_output():
    with pytest.raises(ValueError, match="--selection-output"):
        main(["final", "--stage", "select"])


def test_final_select_requires_two_candidates(tmp_path: Path):
    with pytest.raises(ValueError, match="at least two"):
        main(
            [
                "final",
                "--stage",
                "select",
                "--selection-output",
                str(tmp_path / "s.json"),
                "--candidate",
                "a:x",
            ]
        )
    with pytest.raises(ValueError, match="at least two"):
        main(
            [
                "final",
                "--stage",
                "select",
                "--selection-output",
                str(tmp_path / "s.json"),
            ]
        )


def test_final_select_refuses_existing_selection_output(tmp_path: Path):
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")

    with pytest.raises(FileExistsError, match="selection report"):
        main(
            [
                "final",
                "--stage",
                "select",
                "--selection-output",
                str(selection_path),
                "--candidate",
                "a:missing-a.pkl",
                "--candidate",
                "b:missing-b.pkl",
            ]
        )


def test_final_evaluate_requires_existing_selection(tmp_path: Path):
    with pytest.raises(ValueError, match="selection report"):
        main(
            [
                "final",
                "--stage",
                "evaluate",
                "--selection-output",
                str(tmp_path / "missing.json"),
                "--evaluation-output",
                str(tmp_path / "evaluation.json"),
                "--apriori-checkpoint",
                "missing.pkl",
            ]
        )


def test_final_evaluate_requires_evaluation_output_and_apriori_checkpoint(
    tmp_path: Path,
):
    selection = _synthetic_selection_report(tmp_path)

    with pytest.raises(ValueError, match="--evaluation-output"):
        main(
            [
                "final",
                "--stage",
                "evaluate",
                "--selection-output",
                str(selection),
                "--apriori-checkpoint",
                "missing.pkl",
            ]
        )
    with pytest.raises(ValueError, match="--apriori-checkpoint"):
        main(
            [
                "final",
                "--stage",
                "evaluate",
                "--selection-output",
                str(selection),
                "--evaluation-output",
                str(tmp_path / "evaluation.json"),
            ]
        )


def test_final_evaluate_chain_reaches_checkpoint_loading(tmp_path: Path):
    selection = _synthetic_selection_report(tmp_path)

    with pytest.raises(FileNotFoundError):
        main(
            [
                "final",
                "--stage",
                "evaluate",
                "--selection-output",
                str(selection),
                "--evaluation-output",
                str(tmp_path / "evaluation.json"),
                "--apriori-checkpoint",
                str(tmp_path / "missing-apriori.pkl"),
                "--split",
                "validation",
            ]
        )


def test_final_apriori_chain_refuses_existing_output_dir(tmp_path: Path):
    selection = _synthetic_selection_report(tmp_path)
    output_dir = tmp_path / "apriori-out"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="a-priori output directory"):
        main(
            [
                "final",
                "--stage",
                "apriori",
                "--selection-output",
                str(selection),
                "--apriori-output-dir",
                str(output_dir),
            ]
        )


def test_final_apriori_chain_derives_configs_from_reference(tmp_path: Path):
    selection = _synthetic_selection_report(tmp_path)

    with pytest.raises(FileNotFoundError):
        main(
            [
                "final",
                "--stage",
                "apriori",
                "--selection-output",
                str(selection),
                "--apriori-output-dir",
                str(tmp_path / "apriori-out"),
            ]
        )


def test_sanitise_json_encodes_nonfinite_explicitly(tmp_path: Path):
    payload = {
        "positive": float("inf"),
        "negative": float("-inf"),
        "nan": float("nan"),
        "nested": {"value": 1.0, "bad": float("inf")},
        "items": [2.0, float("nan")],
        "whole": 3,
    }

    sanitised = _sanitise_json(payload)
    assert sanitised["positive"] == "Infinity"
    assert sanitised["negative"] == "-Infinity"
    assert sanitised["nan"] == "NaN"
    assert sanitised["nested"]["bad"] == "Infinity"
    assert sanitised["items"] == [2.0, "NaN"]
    assert sanitised["nested"]["value"] == 1.0
    assert sanitised["whole"] == 3

    path = write_report_refusing_existing(tmp_path / "nonfinite.json", payload)
    encoded = json.loads(path.read_text())
    assert encoded["positive"] == "Infinity"
    assert encoded["negative"] == "-Infinity"
    assert encoded["nan"] == "NaN"
    assert encoded["items"] == [2.0, "NaN"]


def test_horizons_reject_floats_and_bools():
    with pytest.raises(ValueError, match="integers"):
        _normalise_and_validate_horizons((30.0,))
    with pytest.raises(ValueError, match="integers"):
        _normalise_and_validate_horizons((True, 30))


def test_model_selection_rejects_float_horizon(tmp_path: Path):
    with pytest.raises(ValueError, match="integer"):
        run_model_selection(
            {"a": "x", "b": "y"},
            output_path=tmp_path / "selection.json",
            horizon=30.0,
        )


def test_train_matched_apriori_rejects_float_updates(tmp_path: Path):
    with pytest.raises(ValueError, match="integer"):
        train_matched_apriori_baseline(700.0, tmp_path / "never")


@pytest.fixture
def tiny_parameter_count(monkeypatch):
    monkeypatch.setattr(final_eval_module, "parameter_count", lambda: 4)
    return 4


def _candidate_checkpoint(path: Path, params: jax.Array, updates: int) -> Path:
    return save_training_checkpoint(
        path,
        {
            "params_flat": params,
            "completed_updates": updates,
            "completed_unroll": 30,
            "losses": [1.0],
            "solver_config": asdict(SolverConfig(dt=0.002)),
            "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=20.0)),
        },
    )


def test_selection_generates_each_validation_reference_once(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    reference_calls = {"count": 0}
    loss_calls = {"count": 0}

    def fake_reference(seed, num_steps, *, split, config):
        reference_calls["count"] += 1
        assert split == "validation"
        return ReferenceTrajectory(
            initial_coarse=jnp.zeros((1, 2, 2), dtype=jnp.float32),
            targets=jnp.zeros((num_steps, 1, 2, 2), dtype=jnp.float32),
        )

    def fake_loss(_stepper, params, _initial, _targets):
        loss_calls["count"] += 1
        return jnp.asarray(params[0], dtype=jnp.float32)

    monkeypatch.setattr(
        final_eval_module, "generate_reference_trajectory", fake_reference
    )
    monkeypatch.setattr(final_eval_module, "aposteriori_loss", fake_loss)

    selection_path = tmp_path / "selection.json"
    low_path = _candidate_checkpoint(
        tmp_path / "low.pkl", jnp.full((4,), 1.0, dtype=jnp.float32), 300
    )
    high_path = _candidate_checkpoint(
        tmp_path / "high.pkl", jnp.full((4,), 2.0, dtype=jnp.float32), 700
    )
    low_rel, high_rel = os.path.relpath(low_path), os.path.relpath(high_path)
    report = run_model_selection(
        {"low": low_rel, "high": high_rel},
        output_path=selection_path,
    )

    assert reference_calls["count"] == 32
    assert loss_calls["count"] == 64
    assert report["selected"]["name"] == "low"
    assert report["selected"]["mean_vorticity_mse"] == 1.0
    assert report["candidates"]["high"]["mean_vorticity_mse"] == 2.0
    assert report["num_seeds"] == 32
    assert report["seed_range"] == "10000..10031"
    assert report["horizon"] == SELECTION_HORIZON
    assert report["split"] == "validation"
    assert report["protocol"]["name"] == "final-submission-evaluation"
    assert report["protocol"]["version"] == 1
    assert report["warnings"] == []
    assert "32" in report["criterion"] and "30" in report["criterion"]

    persisted = json.loads(selection_path.read_text())
    assert persisted["selected"]["name"] == "low"
    assert persisted["candidates"]["low"]["checkpoint"] == low_rel
    assert persisted["candidates"]["high"]["checkpoint"] == high_rel
    assert not Path(persisted["candidates"]["low"]["checkpoint"]).is_absolute()
    assert len(persisted["per_seed_errors"]["low"]) == 32
    assert len(persisted["per_seed_errors"]["high"]) == 32


def test_validate_selection_report_fail_closed(tmp_path: Path):
    report = json.loads(_synthetic_selection_report(tmp_path).read_text())

    validate_selection_report(report)

    def with_selected_name(rpt, name):
        shallow = dict(rpt)
        shallow["selected"] = {**shallow["selected"], "name": name}
        return shallow

    def with_means(rpt, winner_mean, loser_mean, *, selected_mean=None):
        shallow = dict(rpt)
        shallow["candidates"] = {
            "winner": {
                **rpt["candidates"]["winner"],
                "mean_vorticity_mse": winner_mean,
            },
            "loser": {
                **rpt["candidates"]["loser"],
                "mean_vorticity_mse": loser_mean,
            },
        }
        selected = dict(shallow["selected"])
        selected["mean_vorticity_mse"] = (
            winner_mean if selected_mean is None else selected_mean
        )
        shallow["selected"] = selected
        return shallow

    def with_only_winner(rpt):
        shallow = dict(rpt)
        shallow["candidates"] = {"winner": rpt["candidates"]["winner"]}
        return shallow

    overrides = [
        ({"horizon": 5}, "horizon"),
        ({"split": "test"}, "split"),
        ({"num_seeds": 1}, "num_seeds"),
        ({"seed_range": "0..1"}, "seed_range"),
        ({"criterion": "tampered criterion"}, "criterion"),
        ({"protocol": {**PROTOCOL, "name": "other"}}, "protocol name"),
        ({"protocol": {**PROTOCOL, "version": 2}}, "version"),
        (
            {
                "protocol": {
                    **PROTOCOL,
                    "locks": {**PROTOCOL["locks"], "selection_horizon": 5},
                }
            },
            "locks",
        ),
        (with_only_winner(report), "at least two"),
        ({"candidates": {}}, "at least two"),
        (with_means(report, "banana", 2.0), "mean_vorticity_mse"),
        (with_means(report, 2.0, 1.0), "lowest"),
        (with_selected_name(report, "loser"), "lowest"),
        (with_means(report, 1.0, 2.0, selected_mean=1.5), "must equal"),
        (with_means(report, "Infinity", "Infinity"), "finite"),
    ]
    for override, message in overrides:
        with pytest.raises(ValueError) as excinfo:
            validate_selection_report({**report, **override})
        assert "selection report" in str(excinfo.value)
        assert message in str(excinfo.value)


def test_evaluation_test_split_enforces_locked_horizons(tmp_path: Path):
    selection = _synthetic_selection_report(tmp_path)

    with pytest.raises(ValueError, match="locked to horizons"):
        run_evaluation_stage(
            selection_report_path=selection,
            apriori_checkpoint_path=tmp_path / "missing-apriori.pkl",
            output_path=tmp_path / "evaluation.json",
            split="test",
            horizons=(30,),
        )


def test_evaluation_test_split_validates_selection_before_generation(
    tmp_path: Path,
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("test data generation reached before validation")

    monkeypatch.setattr(final_eval_module, "generate_reference_trajectory", forbidden)
    invalid = _synthetic_selection_report(tmp_path, horizon=5)

    with pytest.raises(ValueError, match="horizon"):
        run_evaluation_stage(
            selection_report_path=invalid,
            apriori_checkpoint_path=tmp_path / "missing-apriori.pkl",
            output_path=tmp_path / "evaluation.json",
            split="test",
        )


def _forbid_test_generation(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("test data generation reached before validation")

    monkeypatch.setattr(final_eval_module, "generate_reference_trajectory", forbidden)


def test_evaluation_rejects_apriori_updates_mismatch_before_generation(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    _forbid_test_generation(monkeypatch)
    selected = _candidate_checkpoint(
        tmp_path / "selected.pkl", jnp.arange(4, dtype=jnp.float32), 700
    )
    selection = _synthetic_selection_report(tmp_path, selected)
    apriori = save_training_checkpoint(
        tmp_path / "apriori-300.pkl",
        {
            "params_flat": jnp.arange(4, dtype=jnp.float32),
            "training_scheme": APRIORI_SCHEME,
            "completed_updates": 300,
            "solver_config": asdict(SolverConfig(dt=0.002)),
            "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=20.0)),
        },
    )

    with pytest.raises(ValueError, match="700"):
        run_evaluation_stage(
            selection_report_path=selection,
            apriori_checkpoint_path=apriori,
            output_path=tmp_path / "evaluation.json",
            split="test",
        )


def test_evaluation_rejects_apriori_config_mismatch_before_generation(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    _forbid_test_generation(monkeypatch)
    selected = _candidate_checkpoint(
        tmp_path / "selected.pkl", jnp.arange(4, dtype=jnp.float32), 700
    )
    selection = _synthetic_selection_report(tmp_path, selected)
    apriori = save_training_checkpoint(
        tmp_path / "apriori-mismatch.pkl",
        {
            "params_flat": jnp.arange(4, dtype=jnp.float32),
            "training_scheme": APRIORI_SCHEME,
            "completed_updates": 700,
            "solver_config": asdict(SolverConfig(dt=0.002)),
            "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=10.0)),
        },
    )

    with pytest.raises(ValueError, match="configs"):
        run_evaluation_stage(
            selection_report_path=selection,
            apriori_checkpoint_path=apriori,
            output_path=tmp_path / "evaluation.json",
            split="test",
        )


def _mock_chain(monkeypatch, tmp_path: Path, extra_args: Sequence[str] = ()) -> dict:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")
    calls: dict[str, tuple] = {}

    def fake_select(candidates, *, output_path, horizon):
        calls["select"] = (dict(candidates), str(output_path), horizon)
        return {
            "candidates": {"a": {"checkpoint": "runs/selected-a.pkl"}},
            "selected": {"name": "a", "mean_vorticity_mse": 1.5},
            "criterion": "lowest full-32-seed validation vorticity-MSE at horizon 30",
            "horizon": 30,
            "num_seeds": 32,
        }

    def fake_apriori(num_updates, output_dir, *, reference_checkpoint):
        calls["apriori"] = (num_updates, str(output_dir), reference_checkpoint)
        return {
            "checkpoint": "runs/apriori/checkpoint.pkl",
            "summary": "runs/apriori/training-summary.json",
            "updates": 700,
            "first_loss": 1.0,
            "final_loss": 0.5,
        }

    def fake_evaluate(
        *,
        selection_report_path,
        apriori_checkpoint_path,
        output_path,
        split,
        horizons,
    ):
        calls["evaluate"] = (
            str(selection_report_path),
            apriori_checkpoint_path,
            str(output_path),
            split,
            tuple(horizons),
        )
        return {"methods": {}, "per_seed_errors": {}}

    monkeypatch.setattr(cli_module, "run_model_selection", fake_select)
    monkeypatch.setattr(cli_module, "train_matched_apriori_baseline", fake_apriori)
    monkeypatch.setattr(cli_module, "run_evaluation_stage", fake_evaluate)

    exit_code = main(
        [
            "final",
            "--selection-output",
            str(selection_path),
            "--candidate",
            "a:runs/a.pkl",
            "--candidate",
            "b:runs/b.pkl",
            "--apriori-output-dir",
            str(tmp_path / "apriori-out"),
            "--evaluation-output",
            str(tmp_path / "evaluation.json"),
            *extra_args,
        ]
    )
    calls["exit_code"] = exit_code
    return calls


def test_final_cli_full_chain_with_mocks(monkeypatch, tmp_path: Path):
    calls = _mock_chain(monkeypatch, tmp_path)

    assert calls["exit_code"] == 0
    candidate_names, output, horizon = calls["select"]
    assert set(candidate_names) == {"a", "b"}
    assert output == str(tmp_path / "selection.json")
    assert horizon == SELECTION_HORIZON

    updates, output_dir, reference = calls["apriori"]
    assert updates == MATCHED_APRIORI_UPDATES
    assert output_dir == str(tmp_path / "apriori-out")
    assert reference == "runs/selected-a.pkl"

    selection_report_path, apriori_checkpoint, evaluation_output, split, horizons = (
        calls["evaluate"]
    )
    assert selection_report_path == str(tmp_path / "selection.json")
    assert apriori_checkpoint == "runs/apriori/checkpoint.pkl"
    assert evaluation_output == str(tmp_path / "evaluation.json")
    assert split == "test"
    assert horizons == FINAL_EVALUATION_HORIZONS


def test_final_cli_explicit_apriori_checkpoint_overrides_chain(
    monkeypatch,
    tmp_path: Path,
):
    calls = _mock_chain(
        monkeypatch,
        tmp_path,
        extra_args=["--apriori-checkpoint", "runs/explicit-apriori.pkl"],
    )

    assert calls["apriori"][0] == MATCHED_APRIORI_UPDATES
    assert calls["evaluate"][1] == "runs/explicit-apriori.pkl"


def test_final_rejects_apriori_reference_knob():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["final", "--apriori-reference-checkpoint", "some.pkl"]
        )


def test_final_apriori_validates_existing_selection_before_training(
    tmp_path: Path,
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("baseline training reached before selection validation")

    monkeypatch.setattr(cli_module, "train_matched_apriori_baseline", forbidden)
    tampered = _synthetic_selection_report(tmp_path, criterion="tampered criterion")

    with pytest.raises(ValueError, match="selection report"):
        main(
            [
                "final",
                "--stage",
                "apriori",
                "--selection-output",
                str(tampered),
                "--apriori-output-dir",
                str(tmp_path / "apriori-out"),
            ]
        )


def test_final_evaluate_validates_existing_selection_before_evaluation(
    tmp_path: Path,
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("evaluation reached before selection validation")

    monkeypatch.setattr(cli_module, "run_evaluation_stage", forbidden)
    tampered = _synthetic_selection_report(tmp_path, criterion="tampered criterion")

    with pytest.raises(ValueError, match="selection report"):
        main(
            [
                "final",
                "--stage",
                "evaluate",
                "--selection-output",
                str(tampered),
                "--evaluation-output",
                str(tmp_path / "evaluation.json"),
                "--apriori-checkpoint",
                "missing.pkl",
            ]
        )


def test_final_apriori_derives_reference_from_selected_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    selection = _synthetic_selection_report(tmp_path)
    recorded: dict[str, str] = {}

    def fake_apriori(num_updates, output_dir, *, reference_checkpoint):
        recorded["reference"] = reference_checkpoint
        return {
            "checkpoint": "runs/apriori/checkpoint.pkl",
            "summary": "runs/apriori/training-summary.json",
            "updates": 700,
            "first_loss": 1.0,
            "final_loss": 0.5,
        }

    monkeypatch.setattr(cli_module, "train_matched_apriori_baseline", fake_apriori)

    exit_code = main(
        [
            "final",
            "--stage",
            "apriori",
            "--selection-output",
            str(selection),
            "--apriori-output-dir",
            str(tmp_path / "apriori-out"),
        ]
    )

    stored = json.loads(selection.read_text())
    assert exit_code == 0
    assert recorded["reference"] == stored["candidates"]["winner"]["checkpoint"]


def test_load_candidate_params_rejects_wrong_size(
    tmp_path: Path,
    tiny_parameter_count,
):
    path = save_training_checkpoint(
        tmp_path / "wrong-size.pkl",
        {
            "params_flat": jnp.arange(5, dtype=jnp.float32),
            "completed_updates": 1,
            "solver_config": asdict(SolverConfig(dt=0.002)),
        },
    )

    with pytest.raises(ValueError, match="exactly"):
        load_candidate_params(path)


def test_train_matched_apriori_rejects_wrong_param_size_before_writing(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    monkeypatch.setattr(
        final_eval_module,
        "train_apriori_baseline",
        lambda num_updates, *, solver_config, dns_config: (
            jnp.zeros((3,), dtype=jnp.float32),
            (1.0, 2.0),
        ),
    )
    output_dir = tmp_path / "apriori-out"

    with pytest.raises(ValueError, match="expected"):
        train_matched_apriori_baseline(
            2,
            output_dir,
            solver_config=SolverConfig(dt=0.002),
            dns_config=DNSConfig(dt=0.002, vorticity_amplitude=20.0),
        )
    assert not output_dir.exists()


def test_train_matched_apriori_rejects_nonfinite_params_before_writing(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    params = jnp.asarray([1.0, math.nan, 2.0, 3.0], dtype=jnp.float32)
    monkeypatch.setattr(
        final_eval_module,
        "train_apriori_baseline",
        lambda num_updates, *, solver_config, dns_config: (params, (1.0, 2.0)),
    )
    output_dir = tmp_path / "apriori-out"

    with pytest.raises(FloatingPointError, match="non-finite parameters"):
        train_matched_apriori_baseline(
            2,
            output_dir,
            solver_config=SolverConfig(dt=0.002),
            dns_config=DNSConfig(dt=0.002, vorticity_amplitude=20.0),
        )
    assert not output_dir.exists()


def test_train_matched_apriori_rejects_loss_count_mismatch_before_writing(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    monkeypatch.setattr(
        final_eval_module,
        "train_apriori_baseline",
        lambda num_updates, *, solver_config, dns_config: (
            jnp.arange(4, dtype=jnp.float32),
            (1.0,),
        ),
    )
    output_dir = tmp_path / "apriori-out"

    with pytest.raises(ValueError, match="losses"):
        train_matched_apriori_baseline(
            2,
            output_dir,
            solver_config=SolverConfig(dt=0.002),
            dns_config=DNSConfig(dt=0.002, vorticity_amplitude=20.0),
        )
    assert not output_dir.exists()


def test_train_matched_apriori_rejects_nonfinite_losses_before_writing(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    monkeypatch.setattr(
        final_eval_module,
        "train_apriori_baseline",
        lambda num_updates, *, solver_config, dns_config: (
            jnp.arange(4, dtype=jnp.float32),
            (1.0, math.inf),
        ),
    )
    output_dir = tmp_path / "apriori-out"

    with pytest.raises(FloatingPointError, match="non-finite losses"):
        train_matched_apriori_baseline(
            2,
            output_dir,
            solver_config=SolverConfig(dt=0.002),
            dns_config=DNSConfig(dt=0.002, vorticity_amplitude=20.0),
        )
    assert not output_dir.exists()


def test_train_matched_apriori_mocked_happy_path_writes_artefacts(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    params = jnp.arange(4, dtype=jnp.float32)
    monkeypatch.setattr(
        final_eval_module,
        "train_apriori_baseline",
        lambda num_updates, *, solver_config, dns_config: (params, (1.0, 2.0)),
    )
    output_dir = tmp_path / "apriori-out"

    result = train_matched_apriori_baseline(
        2,
        output_dir,
        solver_config=SolverConfig(dt=0.002),
        dns_config=DNSConfig(dt=0.002, vorticity_amplitude=20.0),
    )

    assert result["updates"] == 2
    checkpoint = load_training_checkpoint(result["checkpoint"])
    assert checkpoint["training_scheme"] == APRIORI_SCHEME
    assert checkpoint["completed_updates"] == 2
    assert checkpoint["losses"] == [1.0, 2.0]
    assert jnp.array_equal(checkpoint["params_flat"], params)
    summary = json.loads((output_dir / "training-summary.json").read_text())
    assert summary["updates"] == 2


def test_evaluation_report_preserves_relative_paths_and_method_metadata(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    selected_path = _candidate_checkpoint(
        tmp_path / "selected.pkl", jnp.arange(4, dtype=jnp.float32), 700
    )
    apriori_path = save_training_checkpoint(
        tmp_path / "apriori.pkl",
        {
            "params_flat": jnp.arange(4, dtype=jnp.float32),
            "training_scheme": APRIORI_SCHEME,
            "completed_updates": 700,
            "solver_config": asdict(SolverConfig(dt=0.002)),
            "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=20.0)),
        },
    )
    selection_rel = os.path.relpath(selected_path)
    apriori_rel = os.path.relpath(apriori_path)
    selection_report = _synthetic_selection_report(tmp_path, selection_rel)
    evaluation_rel = os.path.relpath(tmp_path / "evaluation.json")

    def fake_reference(seed, num_steps, *, split, config):
        return ReferenceTrajectory(
            initial_coarse=jnp.zeros((1, 2, 2), dtype=jnp.float32),
            targets=jnp.zeros((num_steps, 1, 2, 2), dtype=jnp.float32),
        )

    def fake_closure(stepper, params_flat, state, num_steps):
        return jnp.zeros((num_steps, 1, 2, 2), dtype=jnp.float32)

    def fake_no_closure(stepper, state, num_steps):
        return jnp.zeros((num_steps, 1, 2, 2), dtype=jnp.float32)

    def fake_smagorinsky(stepper, state, num_steps, *, dynamic):
        return jnp.zeros((num_steps, 1, 2, 2), dtype=jnp.float32)

    monkeypatch.setattr(
        final_eval_module, "generate_reference_trajectory", fake_reference
    )
    monkeypatch.setattr(final_eval_module, "closure_rollout", fake_closure)
    monkeypatch.setattr(final_eval_module, "no_closure_rollout", fake_no_closure)
    monkeypatch.setattr(final_eval_module, "smagorinsky_rollout", fake_smagorinsky)

    report = run_evaluation_stage(
        selection_report_path=str(selection_report),
        apriori_checkpoint_path=apriori_rel,
        output_path=evaluation_rel,
        split="validation",
        horizons=(30,),
    )

    assert report["selected_aposteriori_checkpoint"] == selection_rel
    assert report["apriori_checkpoint"] == apriori_rel
    assert report["methods"]["aposteriori-selected"]["checkpoint"] == selection_rel
    assert report["methods"]["apriori-matched"]["checkpoint"] == apriori_rel
    assert report["methods"]["apriori-matched"]["completed_updates"] == 700
    assert report["methods"]["apriori-matched"]["configs_match_selected"] is True
    assert (
        report["methods"]["static-smagorinsky"]["coefficient"]
        == STATIC_SMAGORINSKY_COEFFICIENT
    )
    assert (
        report["methods"]["dynamic-smagorinsky"]["test_filter_ratio"]
        == SMAGORINSKY_TEST_FILTER_RATIO
    )
    assert not Path(report["apriori_checkpoint"]).is_absolute()
    assert not Path(report["selected_aposteriori_checkpoint"]).is_absolute()
    assert (
        report["methods"]["aposteriori-selected"]["sha256"]
        == hashlib.sha256(Path(selection_rel).read_bytes()).hexdigest()
    )
    assert (
        report["methods"]["apriori-matched"]["sha256"]
        == hashlib.sha256(Path(apriori_rel).read_bytes()).hexdigest()
    )


def test_selection_records_sha256_bound_to_loaded_bytes(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    def fake_reference(seed, num_steps, *, split, config):
        return ReferenceTrajectory(
            initial_coarse=jnp.zeros((1, 2, 2), dtype=jnp.float32),
            targets=jnp.zeros((num_steps, 1, 2, 2), dtype=jnp.float32),
        )

    def fake_loss(_stepper, params, _initial, _targets):
        return jnp.asarray(params[0], dtype=jnp.float32)

    monkeypatch.setattr(
        final_eval_module, "generate_reference_trajectory", fake_reference
    )
    monkeypatch.setattr(final_eval_module, "aposteriori_loss", fake_loss)

    low_path = _candidate_checkpoint(
        tmp_path / "low.pkl", jnp.full((4,), 1.0, dtype=jnp.float32), 300
    )
    high_path = _candidate_checkpoint(
        tmp_path / "high.pkl", jnp.full((4,), 2.0, dtype=jnp.float32), 700
    )
    selection_path = tmp_path / "selection.json"
    run_model_selection(
        {"low": low_path, "high": high_path},
        output_path=selection_path,
    )

    persisted = json.loads(selection_path.read_text())
    expected = {
        "low": hashlib.sha256(low_path.read_bytes()).hexdigest(),
        "high": hashlib.sha256(high_path.read_bytes()).hexdigest(),
    }
    for name, digest in expected.items():
        assert persisted["candidates"][name]["sha256"] == digest
    assert (
        persisted["candidates"][persisted["selected"]["name"]]["sha256"]
        == expected[persisted["selected"]["name"]]
    )

    # The recorded digest binds the exact bytes that produced the selected
    # parameters: replacing the file invalidates the binding.
    high_path.write_bytes(b"tampered")
    assert hashlib.sha256(high_path.read_bytes()).hexdigest() != expected["high"]


def test_load_candidate_params_with_digest_verifies_and_records(
    tmp_path: Path,
    tiny_parameter_count,
):
    path = _candidate_checkpoint(
        tmp_path / "candidate.pkl", jnp.arange(4, dtype=jnp.float32), 700
    )
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    checkpoint, params, digest = load_candidate_params_with_digest(
        path, expected_sha256=expected
    )
    assert digest == expected
    assert jnp.array_equal(params, jnp.arange(4, dtype=jnp.float32))
    assert checkpoint["completed_updates"] == 700

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_candidate_params_with_digest(path, expected_sha256="1" * 64)
    with pytest.raises(ValueError, match="SHA-256 digest"):
        load_candidate_params_with_digest(path, expected_sha256="not-a-digest")


def test_evaluation_rejects_missing_digest_for_test_split_before_generation(
    tmp_path: Path,
    monkeypatch,
):
    _forbid_test_generation(monkeypatch)
    selection = _synthetic_selection_report(tmp_path)
    report = json.loads(selection.read_text())
    for candidate in report["candidates"].values():
        del candidate["sha256"]
    selection.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="sha256"):
        run_evaluation_stage(
            selection_report_path=selection,
            apriori_checkpoint_path=tmp_path / "missing-apriori.pkl",
            output_path=tmp_path / "evaluation.json",
            split="test",
        )


def test_evaluation_rejects_replaced_selected_checkpoint_before_generation(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    _forbid_test_generation(monkeypatch)
    selected = _candidate_checkpoint(
        tmp_path / "selected.pkl", jnp.arange(4, dtype=jnp.float32), 700
    )
    selection = _synthetic_selection_report(tmp_path, selected)
    # Replace the checkpoint at the recorded path with different bytes.
    selected.write_bytes(b"replaced with a different checkpoint")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_evaluation_stage(
            selection_report_path=selection,
            apriori_checkpoint_path=tmp_path / "missing-apriori.pkl",
            output_path=tmp_path / "evaluation.json",
            split="test",
        )


def test_evaluation_rejects_swapped_digest_before_generation(
    tmp_path: Path,
    monkeypatch,
    tiny_parameter_count,
):
    _forbid_test_generation(monkeypatch)
    selected = _candidate_checkpoint(
        tmp_path / "selected.pkl", jnp.arange(4, dtype=jnp.float32), 700
    )
    selection = _synthetic_selection_report(tmp_path, selected)
    report = json.loads(selection.read_text())
    for candidate in report["candidates"].values():
        candidate["sha256"] = "1" * 64
    selection.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_evaluation_stage(
            selection_report_path=selection,
            apriori_checkpoint_path=tmp_path / "missing-apriori.pkl",
            output_path=tmp_path / "evaluation.json",
            split="test",
        )


def test_legacy_structure_validation_accepts_digestless_reports(tmp_path: Path):
    selection = _synthetic_selection_report(tmp_path)
    report = json.loads(selection.read_text())
    for candidate in report["candidates"].values():
        del candidate["sha256"]
    legacy = tmp_path / "legacy-selection.json"
    legacy.write_text(json.dumps(report))
    loaded = json.loads(legacy.read_text())

    # The explicitly named legacy mode accepts a pre-digest report...
    validate_selection_report_legacy_structure(loaded)
    # ...and still rejects tampering and malformed digests when present.
    with pytest.raises(ValueError, match="criterion"):
        validate_selection_report_legacy_structure(
            json.loads(
                _synthetic_selection_report(
                    tmp_path, criterion="tampered criterion"
                ).read_text()
            )
        )
    digest_carrying = json.loads(_synthetic_selection_report(tmp_path).read_text())
    digest_carrying["candidates"]["winner"]["sha256"] = "nonsense"
    with pytest.raises(ValueError, match="SHA-256 digest"):
        validate_selection_report_legacy_structure(digest_carrying)

    # The strict validator used by test evaluation rejects the same report.
    with pytest.raises(ValueError, match="sha256"):
        validate_selection_report(loaded)


def test_cli_evaluate_on_test_rejects_digestless_selection_report(tmp_path: Path):
    selection = _synthetic_selection_report(tmp_path)
    report = json.loads(selection.read_text())
    for candidate in report["candidates"].values():
        del candidate["sha256"]
    selection.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="selection report"):
        main(
            [
                "final",
                "--stage",
                "evaluate",
                "--selection-output",
                str(selection),
                "--evaluation-output",
                str(tmp_path / "evaluation.json"),
                "--apriori-checkpoint",
                str(tmp_path / "missing-apriori.pkl"),
            ]
        )
