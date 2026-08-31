from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_hybrid_closure.checkpointing import save_training_checkpoint
from tesseract_hybrid_closure.closure import initial_parameters
from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
from tesseract_hybrid_closure.constants import (
    TEST_SEED_RANGE,
    VALIDATION_SEED_RANGE,
)
from tesseract_hybrid_closure.solver import CoarseVorticityStepper
from tesseract_hybrid_closure.spectral_diagnostics import (
    DIAGNOSTIC_SPLIT,
    REFERENCE_METHOD,
    build_methods,
    run_spectral_diagnostic,
)


@pytest.fixture
def checkpoint_path(tmp_path):
    return save_training_checkpoint(
        tmp_path / "checkpoint.pkl",
        {
            "params_flat": jnp.asarray(initial_parameters()),
            "completed_updates": 700,
            "solver_config": asdict(SolverConfig(dt=0.002)),
            "dns_config": asdict(DNSConfig(dt=0.002, vorticity_amplitude=20.0)),
        },
    )


def test_the_test_split_is_refused(checkpoint_path):
    with pytest.raises(ValueError, match="sealed test split stays sealed"):
        run_spectral_diagnostic(
            aposteriori_checkpoint=checkpoint_path,
            apriori_checkpoint=None,
            seeds=[TEST_SEED_RANGE[0]],
            report_steps=[1],
            split="test",
        )


def test_seeds_outside_the_validation_split_are_refused(checkpoint_path):
    with pytest.raises(ValueError, match="outside the 'validation' split"):
        run_spectral_diagnostic(
            aposteriori_checkpoint=checkpoint_path,
            apriori_checkpoint=None,
            seeds=[TEST_SEED_RANGE[0]],
            report_steps=[1],
        )


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([], "at least one report step"),
        ([0], "must be positive"),
        ([1, 1], "must be unique"),
    ],
)
def test_invalid_report_steps_are_refused(checkpoint_path, steps, message):
    with pytest.raises(ValueError, match=message):
        run_spectral_diagnostic(
            aposteriori_checkpoint=checkpoint_path,
            apriori_checkpoint=None,
            seeds=[VALIDATION_SEED_RANGE[0]],
            report_steps=steps,
        )


def test_duplicate_seeds_are_refused(checkpoint_path):
    seed = VALIDATION_SEED_RANGE[0]
    with pytest.raises(ValueError, match="seeds must be unique"):
        run_spectral_diagnostic(
            aposteriori_checkpoint=checkpoint_path,
            apriori_checkpoint=None,
            seeds=[seed, seed],
            report_steps=[1],
        )


def test_a_mismatched_apriori_regime_is_refused(tmp_path, checkpoint_path):
    other = save_training_checkpoint(
        tmp_path / "other-regime.pkl",
        {
            "params_flat": jnp.asarray(initial_parameters()),
            "completed_updates": 700,
            "solver_config": asdict(SolverConfig(dt=0.01)),
            "dns_config": asdict(DNSConfig(dt=0.01, vorticity_amplitude=20.0)),
        },
    )

    with pytest.raises(ValueError, match="different regime"):
        run_spectral_diagnostic(
            aposteriori_checkpoint=checkpoint_path,
            apriori_checkpoint=other,
            seeds=[VALIDATION_SEED_RANGE[0]],
            report_steps=[1],
        )


def test_a_checkpoint_without_a_dns_regime_is_refused(tmp_path):
    # The loader already demands solver_config, so a checkpoint missing only
    # the DNS regime is what reaches the diagnostic's own guard.
    destination = save_training_checkpoint(
        tmp_path / "no-dns-regime.pkl",
        {
            "params_flat": jnp.asarray(initial_parameters()),
            "completed_updates": 700,
            "solver_config": asdict(SolverConfig(dt=0.002)),
        },
    )

    with pytest.raises(ValueError, match="refuses to guess the regime"):
        run_spectral_diagnostic(
            aposteriori_checkpoint=destination,
            apriori_checkpoint=None,
            seeds=[VALIDATION_SEED_RANGE[0]],
            report_steps=[1],
        )


def test_a_checkpoint_without_any_provenance_is_refused(tmp_path):
    destination = save_training_checkpoint(
        tmp_path / "regimeless.pkl",
        {
            "params_flat": jnp.asarray(initial_parameters()),
            "completed_updates": 700,
        },
    )

    with pytest.raises(ValueError, match="lacks training provenance"):
        run_spectral_diagnostic(
            aposteriori_checkpoint=destination,
            apriori_checkpoint=None,
            seeds=[VALIDATION_SEED_RANGE[0]],
            report_steps=[1],
        )


def test_report_shape_and_conservation_over_one_short_rollout(checkpoint_path):
    report = run_spectral_diagnostic(
        aposteriori_checkpoint=checkpoint_path,
        apriori_checkpoint=checkpoint_path,
        seeds=[VALIDATION_SEED_RANGE[0]],
        report_steps=[2, 1],
    )

    assert report["split"] == DIAGNOSTIC_SPLIT
    assert report["report_steps"] == [1, 2]
    assert report["wavenumber"][0] == 1
    assert report["wavenumber"][-1] == report["max_wavenumber"] == 32

    methods = set(report["spectra"]["1"])
    assert methods == {REFERENCE_METHOD, "aposteriori", "apriori", "no-closure"}

    for step_spectra in report["spectra"].values():
        for spectra in step_spectra.values():
            for name in ("energy", "enstrophy"):
                values = np.asarray(spectra[name])
                assert values.shape == (32,)
                assert np.all(np.isfinite(values))
                assert np.all(values >= 0.0)

    # Identical checkpoints must give identical spectra, so the a-priori entry
    # here is exactly the a-posteriori one and its distance is zero.
    for step_distances in report["spectral_distance"].values():
        assert step_distances["apriori"] == step_distances["aposteriori"]
        assert step_distances["no-closure"]["energy_log_distance"] > 0.0

    assert report["checkpoints"]["aposteriori"]["sha256"]


def test_smagorinsky_methods_are_added_only_when_requested():
    stepper = CoarseVorticityStepper()
    params = initial_parameters()

    without = build_methods(
        stepper,
        aposteriori_params=params,
        apriori_params=None,
        num_steps=1,
        include_smagorinsky=False,
    )
    with_baselines = build_methods(
        stepper,
        aposteriori_params=params,
        apriori_params=None,
        num_steps=1,
        include_smagorinsky=True,
    )

    assert set(without) == {"aposteriori", "no-closure"}
    assert set(with_baselines) - set(without) == {
        "static-smagorinsky",
        "dynamic-smagorinsky",
    }


def test_decomposition_totals_match_the_requested_endpoint_vorticity_mse(
    checkpoint_path,
):
    from tesseract_hybrid_closure.data import generate_reference_trajectory
    from tesseract_hybrid_closure.losses import no_closure_rollout, vorticity_mse
    from tesseract_hybrid_closure.solver import CoarseVorticityStepper

    seed = VALIDATION_SEED_RANGE[0]
    report = run_spectral_diagnostic(
        aposteriori_checkpoint=checkpoint_path,
        apriori_checkpoint=None,
        seeds=[seed],
        report_steps=[2],
    )

    reference = generate_reference_trajectory(
        seed,
        2,
        split="validation",
        config=DNSConfig(dt=0.002, vorticity_amplitude=20.0),
    )
    stepper = CoarseVorticityStepper(SolverConfig(dt=0.002))
    trajectory = no_closure_rollout(stepper, reference.initial_coarse, 2)
    expected = float(vorticity_mse(trajectory[1], reference.targets[1]))

    decomposition = report["error_decomposition"]["2"]["no-closure"]

    # Every shell is retained, so the terms sum to this requested endpoint's
    # field MSE exactly. The rollout-prefix metric would also average step 1.
    assert decomposition["squared_error_total"] == pytest.approx(expected, rel=1e-5)
    assert decomposition["amplitude_total"] > 0.0
    assert decomposition["phase_total"] > 0.0
    assert 0.0 <= decomposition["phase_fraction"] <= 1.0


def test_the_decomposition_covers_every_method_but_not_the_reference(checkpoint_path):
    report = run_spectral_diagnostic(
        aposteriori_checkpoint=checkpoint_path,
        apriori_checkpoint=checkpoint_path,
        seeds=[VALIDATION_SEED_RANGE[0]],
        report_steps=[1],
    )

    decomposition = report["error_decomposition"]["1"]

    assert set(decomposition) == {"aposteriori", "apriori", "no-closure"}
    assert REFERENCE_METHOD not in decomposition
    for entry in decomposition.values():
        assert len(entry["amplitude"]) == len(report["decomposition_wavenumber"])
        assert len(entry["phase"]) == len(report["decomposition_wavenumber"])
        assert all(value >= 0.0 for value in entry["amplitude"])
        assert all(value >= 0.0 for value in entry["phase"])
