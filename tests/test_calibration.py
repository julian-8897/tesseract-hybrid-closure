import pytest

from tesseract_hybrid_closure.calibration import (
    calibrate_regimes,
    strongest_finite_regime,
)


def test_calibration_reports_and_selects_largest_finite_gap():
    results = calibrate_regimes(
        amplitudes=(1.0, 2.0),
        timesteps=(1.0e-3,),
        seeds=(10_000,),
        num_steps=2,
    )
    selected = strongest_finite_regime(results)

    assert len(results) == 2
    assert all(result.finite for result in results)
    assert all(result.num_trajectories == 1 for result in results)
    assert selected.mean_no_closure_mse == max(
        result.mean_no_closure_mse for result in results
    )


def test_calibration_rejects_empty_scan():
    with pytest.raises(ValueError, match="non-empty"):
        calibrate_regimes(amplitudes=(), timesteps=(1.0e-3,), seeds=(10_000,))
