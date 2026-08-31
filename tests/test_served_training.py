import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from tesseract_hybrid_closure import served_training as served_training_module
from tesseract_hybrid_closure.checkpointing import (
    read_training_checkpoint_with_digest,
    save_training_checkpoint,
)
from tesseract_hybrid_closure.closure import initial_parameters
from tesseract_hybrid_closure.constants import LEARNING_RATE
from tesseract_hybrid_closure.served_training import (
    SERVED_VALIDATION_UNROLL,
    ComponentEvidence,
    ServedTrainingResult,
    SourceManifest,
    collect_component_evidence,
    collect_source_manifest,
    preflight_served_training_outputs,
    run_served_training,
    validate_served_training_evidence,
    write_served_training_report,
)

_64_HEX = "a" * 64


def _exploding_reference(*args, **kwargs):
    """Marker: asserting it is called proves DNS generation ran."""
    raise AssertionError("DNS generation ran despite a preflight failure")


def _fake_client(container=None):
    """A client stub with just the introspection surface evidence uses."""

    class _FakeImage:
        def __init__(self, image_id, repo_digests):
            self.id = image_id
            self.attrs = {"RepoDigests": repo_digests}

    class _FakeContainer:
        def __init__(self):
            self.attrs = {}
            self.image = (
                _FakeImage("sha256:" + "c" * 64, ["coarse_solver@sha256:" + _64_HEX])
                if container == "bound"
                else None
            )

    class _FakeClient:
        def __init__(self):
            self.openapi_schema = {
                "openapi": "3.1.0",
                "info": {"title": "stub"},
                "paths": {},
            }
            self._container = _FakeContainer() if container == "bound" else None

        def container_info(self):
            if self._container is None:
                raise RuntimeError("no served container in this stub")
            return self._container

    return _FakeClient()


def _reference_checkpoint(tmp_path):
    """A valid checkpoint whose params are the fixed seeded initialisation."""
    destination = tmp_path / "reference.pkl"
    save_training_checkpoint(
        destination,
        {
            "params_flat": np.asarray(initial_parameters()),
            "completed_updates": 0,
            "solver_config": {"dt": 2e-3},
            "dns_config": {"dt": 2e-3, "vorticity_amplitude": 20.0},
        },
    )
    return destination


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"updates": 0}, ValueError, "updates must be positive"),
        ({"updates": -1}, ValueError, "updates must be positive"),
        ({"updates": True}, TypeError, "updates must be an integer"),
        ({"updates": 1.5}, TypeError, "updates must be an integer"),
        (
            {"updates": 1, "unroll_steps": 0},
            ValueError,
            "unroll_steps must be positive",
        ),
        ({"updates": 1, "unroll_steps": True}, TypeError, "unroll_steps must be"),
        (
            {"updates": 1, "validation_seeds": ()},
            ValueError,
            "at least one validation seed",
        ),
    ],
)
def test_invalid_arguments_are_refused(kwargs, exception, message):
    kwargs.setdefault("use_images", False)
    with pytest.raises(exception, match=message):
        run_served_training(**kwargs)


def test_two_updates_in_process_record_the_composition_evidence():
    # Two validation seeds rather than the default eight: this test is about the
    # boundary evidence, not the score, and scoring dominates the runtime.
    result = run_served_training(
        updates=2, use_images=False, validation_seeds=(10_000, 10_001)
    )

    assert result.updates == 2
    assert result.training_seeds == [0, 1]
    assert result.learning_rate == LEARNING_RATE
    assert result.validation_unroll == SERVED_VALIDATION_UNROLL
    assert len(result.loss_curve) == 2
    assert all(loss > 0.0 for loss in result.loss_curve)
    assert result.gradient_size == 822_977
    assert result.all_gradients_finite
    assert result.parameter_update_norm > 0.0

    # One rollout per update: each contributes its own solver and closure calls,
    # and every update must have driven a solver VJP for the objective to have
    # travelled through the second solver step.
    assert result.solver_vjp_calls == result.updates
    assert result.closure_vjp_calls == result.updates * result.unroll_steps
    assert result.solver_apply_calls == result.updates * result.unroll_steps
    assert result.closure_apply_calls == result.updates * result.unroll_steps
    assert result.solver_vjp_min_cotangent_norm > 0.0
    assert result.closure_vjp_min_cotangent_norm > 0.0


def test_a_short_run_lowers_held_out_validation_error():
    result = run_served_training(
        updates=8, use_images=False, validation_seeds=(10_000, 10_001)
    )

    assert result.validation_mse_after < result.validation_mse_before
    assert result.validation_improved
    # The held-out seeds must not overlap the training seeds.
    assert not set(result.validation_seeds) & set(result.training_seeds)


def test_the_report_round_trips_and_refuses_to_overwrite(tmp_path):
    result = run_served_training(
        updates=1, use_images=False, validation_seeds=(10_000,)
    )
    destination = tmp_path / "served-training.json"

    write_served_training_report(result, destination)
    payload = json.loads(destination.read_text())

    assert payload["updates"] == 1
    assert payload["use_images"] is False
    assert len(payload["loss_curve"]) == 1
    assert "not the submitted model" in payload["note"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_served_training_report(result, destination)


def test_result_is_json_serialisable_without_loss():
    result = ServedTrainingResult(
        updates=1,
        unroll_steps=2,
        training_seeds=[0],
        learning_rate=LEARNING_RATE,
        dt=0.002,
        vorticity_amplitude=20.0,
        use_images=True,
    )

    assert json.loads(json.dumps(result.to_dict()))["updates"] == 1


# ---------------------------------------------------------------------------
# Preflight: outputs are refused before any DNS generation or training.
# ---------------------------------------------------------------------------


def test_preflight_refuses_existing_report_before_any_dns_generation(
    tmp_path, monkeypatch
):
    report = tmp_path / "report.json"
    report.write_text("{}")
    monkeypatch.setattr(
        served_training_module,
        "generate_reference_trajectory",
        _exploding_reference,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite report"):
        run_served_training(
            updates=1,
            use_images=False,
            validation_seeds=(10_000,),
            report_path=report,
        )


def test_preflight_refuses_existing_checkpoint_before_any_dns_generation(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "params.pkl"
    checkpoint.write_bytes(b"existing")
    monkeypatch.setattr(
        served_training_module,
        "generate_reference_trajectory",
        _exploding_reference,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite checkpoint"):
        run_served_training(
            updates=1,
            use_images=False,
            validation_seeds=(10_000,),
            checkpoint_path=checkpoint,
        )


def test_preflight_rejects_identical_report_and_checkpoint_paths(tmp_path):
    with pytest.raises(ValueError, match="must be different files"):
        preflight_served_training_outputs(
            report_path=tmp_path / "same.json",
            checkpoint_path=tmp_path / "same.json",
        )


def test_preflight_accepts_fresh_paths(tmp_path):
    preflight_served_training_outputs(
        report_path=tmp_path / "report.json",
        checkpoint_path=tmp_path / "params.pkl",
    )


# ---------------------------------------------------------------------------
# Persisted evidence: checkpoint, hashes, per-seed scores, manifests.
# ---------------------------------------------------------------------------


def test_run_persists_checkpoint_records_hashes_and_passes_evidence_validation(
    tmp_path,
):
    report = tmp_path / "served-training.json"
    checkpoint = tmp_path / "served-training-params.pkl"
    result = run_served_training(
        updates=1,
        use_images=False,
        validation_seeds=(10_000,),
        report_path=report,
        checkpoint_path=checkpoint,
    )

    # The checkpoint exists, is loadable, and carries the trained parameters.
    payload, digest = read_training_checkpoint_with_digest(checkpoint)
    assert payload["params_flat"].shape == (822_977,)
    assert payload["params_flat"].dtype == np.float32
    assert payload["completed_updates"] == 1
    assert payload["completed_unroll"] == 2
    assert payload["use_images"] is False
    assert payload["losses"] == result.loss_curve
    assert result.checkpoint_sha256 == digest
    assert (
        result.checkpoint_sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )

    # The report carries the checkpoint binding and per-seed scores in
    # reviewable, standards-compliant JSON.
    raw = report.read_text()
    assert raw.startswith("{\n")
    payload_json = json.loads(raw)
    assert payload_json["checkpoint_path"] == str(checkpoint)
    assert payload_json["checkpoint_sha256"] == digest
    assert payload_json["validation_mse_before_per_seed"] == [
        result.validation_mse_before
    ]
    assert payload_json["validation_mse_after_per_seed"] == [
        result.validation_mse_after
    ]

    # The full evidence set validates.
    validate_served_training_evidence(result)


def test_per_seed_validation_and_reference_evidence_are_recorded(tmp_path):
    reference = _reference_checkpoint(tmp_path)
    result = run_served_training(
        updates=1,
        use_images=False,
        validation_seeds=(10_000, 10_001),
        reference_checkpoint=reference,
    )

    assert len(result.validation_mse_before_per_seed) == 2
    assert len(result.validation_mse_after_per_seed) == 2
    assert (
        result.validation_mse_before == sum(result.validation_mse_before_per_seed) / 2
    )
    assert result.validation_mse_after == sum(result.validation_mse_after_per_seed) / 2
    assert result.reference_mse_per_seed == pytest.approx(
        result.validation_mse_before_per_seed
    )
    assert len(result.reference_mse_per_seed) == 2
    assert result.in_process_reference_checkpoint == str(reference)
    assert (
        result.in_process_reference_sha256
        == hashlib.sha256(reference.read_bytes()).hexdigest()
    )
    assert result.in_process_reference_mse == sum(result.reference_mse_per_seed) / 2
    validate_served_training_evidence(result)


def test_local_mode_evidence_is_honest(tmp_path):
    result = run_served_training(
        updates=1,
        use_images=False,
        validation_seeds=(10_000,),
    )

    solver_evidence = result.solver_evidence
    closure_evidence = result.closure_evidence
    assert solver_evidence is not None
    assert closure_evidence is not None
    for evidence in (solver_evidence, closure_evidence):
        assert evidence.mode == "local"
        assert evidence.image_id is None
        assert evidence.repo_digests == ()
        assert evidence.image_reference is None
        assert evidence.config_sha256 and len(evidence.config_sha256) == 64
        assert evidence.schema_sha256 and len(evidence.schema_sha256) == 64
    assert solver_evidence.name == "coarse_solver"
    assert solver_evidence.version == "0.2.0"
    assert solver_evidence.framework == "jax"
    assert closure_evidence.name == "scalar_closure"
    assert closure_evidence.framework == "pytorch"
    assert "in-process clients" in result.note
    assert "not the submitted model" in result.note
    validate_served_training_evidence(result)


def test_source_manifest_records_git_state(tmp_path):
    result = run_served_training(
        updates=1,
        use_images=False,
        validation_seeds=(10_000,),
    )
    manifest = result.source_manifest
    assert len(manifest.git_commit) == 40
    assert int(manifest.git_commit, 16) >= 0
    assert manifest.git_branch
    # The test worktree is dirty while these tests run; the invariant is
    # that dirtiness and the file list agree.
    assert manifest.git_dirty is not None
    if manifest.git_dirty:
        assert manifest.git_dirty_files
    else:
        assert not manifest.git_dirty_files


def test_evidence_validation_rejects_tampered_or_inconsistent_results(tmp_path):
    # A minimal valid result, mirroring what a run records.
    result = ServedTrainingResult(
        updates=1,
        unroll_steps=2,
        training_seeds=[0],
        learning_rate=LEARNING_RATE,
        dt=0.002,
        vorticity_amplitude=20.0,
        use_images=False,
        validation_seeds=[10_000],
        validation_unroll=SERVED_VALIDATION_UNROLL,
        validation_mse_before=1.0,
        validation_mse_after=0.5,
        validation_mse_before_per_seed=[1.0],
        validation_mse_after_per_seed=[0.5],
        solver_evidence=ComponentEvidence(
            name="coarse_solver",
            version="0.2.0",
            framework="jax",
            mode="local",
            config_sha256=_64_HEX,
        ),
        closure_evidence=ComponentEvidence(
            name="scalar_closure",
            version="0.2.0",
            framework="pytorch",
            mode="local",
            config_sha256=_64_HEX,
        ),
        source_manifest=SourceManifest(
            git_commit="a" * 40,
            git_branch="main",
            git_dirty=False,
        ),
        note="test",
    )
    validate_served_training_evidence(result)

    with pytest.raises(ValueError, match="one entry per validation seed"):
        validate_served_training_evidence(
            replace(result, validation_mse_before_per_seed=[1.0, 2.0])
        )
    with pytest.raises(ValueError, match="mean of its per-seed values"):
        validate_served_training_evidence(
            replace(result, validation_mse_before_per_seed=[3.0])
        )
    with pytest.raises(ValueError, match="SHA-256 digest must be"):
        validate_served_training_evidence(
            replace(result, in_process_reference_sha256="deadbeef")
        )
    with pytest.raises(ValueError, match="checkpoint"):
        validate_served_training_evidence(
            replace(
                result,
                checkpoint_path=str(tmp_path / "missing.pkl"),
                checkpoint_sha256=_64_HEX,
            )
        )
    # A valid checkpoint whose recorded digest does not match its bytes.
    reference = _reference_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="does not match the recorded file bytes"):
        validate_served_training_evidence(
            replace(
                result,
                checkpoint_path=str(reference),
                checkpoint_sha256=_64_HEX,
            )
        )
    with pytest.raises(ValueError, match="must bind an image ID"):
        validate_served_training_evidence(
            replace(
                result,
                solver_evidence=ComponentEvidence(
                    name="coarse_solver",
                    version="0.2.0",
                    framework="jax",
                    mode="image",
                    config_sha256=_64_HEX,
                ),
            )
        )
    with pytest.raises(ValueError, match="must not claim container identity"):
        validate_served_training_evidence(
            replace(
                result,
                closure_evidence=ComponentEvidence(
                    name="scalar_closure",
                    version="0.2.0",
                    framework="pytorch",
                    mode="local",
                    image_id="sha256:" + _64_HEX,
                    config_sha256=_64_HEX,
                ),
            )
        )
    with pytest.raises(ValueError, match="git commit must be 40 hex"):
        validate_served_training_evidence(
            replace(
                result,
                source_manifest=SourceManifest(
                    git_commit="abc", git_branch="main", git_dirty=False
                ),
            )
        )
    with pytest.raises(ValueError, match="must record commit and dirty state"):
        validate_served_training_evidence(
            replace(result, source_manifest=SourceManifest())
        )
    with pytest.raises(ValueError, match="must not carry partial git fields"):
        validate_served_training_evidence(
            replace(
                result,
                source_manifest=SourceManifest(
                    git_commit="a" * 40,
                    unavailable_reason="offline",
                ),
            )
        )
    # A manifest that honestly says the source is unknown validates fine.
    validate_served_training_evidence(
        replace(result, source_manifest=SourceManifest(unavailable_reason="offline"))
    )


# ---------------------------------------------------------------------------
# Component evidence collection (unit level, no containers needed).
# ---------------------------------------------------------------------------


def test_image_evidence_binds_live_container_identity():
    evidence = collect_component_evidence(
        _fake_client(container="bound"),
        component="coarse_solver",
        mode="image",
    )

    assert evidence.mode == "image"
    assert evidence.image_reference == "coarse_solver:0.2.0"
    assert evidence.image_id == "sha256:" + "c" * 64
    assert evidence.image_short_id == "sha256:" + "c" * 12
    assert evidence.repo_digests == ("coarse_solver@sha256:" + _64_HEX,)
    assert len(evidence.config_sha256) == 64
    assert len(evidence.schema_sha256) == 64
    assert evidence.framework == "jax"


def test_image_evidence_refuses_a_client_that_cannot_be_bound():
    with pytest.raises(RuntimeError, match="cannot be bound"):
        collect_component_evidence(
            _fake_client(),
            component="coarse_solver",
            mode="image",
        )


def test_local_evidence_never_queries_container_identity():
    evidence = collect_component_evidence(
        _fake_client(),
        component="scalar_closure",
        mode="local",
    )

    assert evidence.mode == "local"
    assert evidence.image_id is None
    assert evidence.repo_digests == ()
    assert evidence.image_reference is None
    assert evidence.name == "scalar_closure"
    assert evidence.framework == "pytorch"


def test_source_manifest_collection_is_consistent():
    manifest = collect_source_manifest()
    if manifest.unavailable_reason is not None:
        assert manifest.git_commit is None
    else:
        assert len(manifest.git_commit) == 40
        assert manifest.git_dirty is not None
