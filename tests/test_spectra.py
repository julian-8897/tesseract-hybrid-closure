import numpy as np
import pytest

from tesseract_hybrid_closure.constants import COARSE_NUM_POINTS, DOMAIN_EXTENT
from tesseract_hybrid_closure.data import band_limited_initial_condition
from tesseract_hybrid_closure.spectra import (
    RadialSpectra,
    coarse_grid_max_wavenumber,
    error_decomposition,
    radial_spectra,
    spectral_distance,
    total_enstrophy,
)


def _single_mode_field(num_points: int, mode_x: int, mode_y: int) -> np.ndarray:
    """cos(kx * x + ky * y) on a 2*pi domain, so its shell is known exactly."""
    axis = 2.0 * np.pi * np.arange(num_points) / num_points
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.cos(mode_x * x + mode_y * y)


def test_enstrophy_spectrum_sums_to_half_mean_square_vorticity():
    rng = np.random.default_rng(0)
    omega = rng.standard_normal((COARSE_NUM_POINTS, COARSE_NUM_POINTS))

    spectra = radial_spectra(omega)

    np.testing.assert_allclose(
        spectra.enstrophy.sum(axis=-1),
        total_enstrophy(omega),
        rtol=1e-12,
    )


def test_single_mode_field_puts_all_enstrophy_in_its_own_shell():
    omega = _single_mode_field(COARSE_NUM_POINTS, 3, 4)

    spectra = radial_spectra(omega)
    shell = int(round(np.hypot(3, 4)))

    assert np.argmax(spectra.enstrophy) == shell
    np.testing.assert_allclose(
        spectra.enstrophy[shell],
        spectra.enstrophy.sum(),
        rtol=1e-12,
    )
    # cos has amplitude 1, so 0.5 * mean(omega**2) = 0.25.
    np.testing.assert_allclose(spectra.enstrophy[shell], 0.25, rtol=1e-12)


def test_axis_aligned_single_mode_obeys_the_per_mode_squared_wavenumber_relation():
    omega = _single_mode_field(COARSE_NUM_POINTS, 5, 0)

    spectra = radial_spectra(omega)

    np.testing.assert_allclose(
        spectra.enstrophy[5],
        25.0 * spectra.energy[5],
        rtol=1e-12,
    )


def test_mean_mode_carries_no_energy():
    omega = np.full((COARSE_NUM_POINTS, COARSE_NUM_POINTS), 3.0)

    spectra = radial_spectra(omega)

    assert spectra.energy[0] == 0.0
    np.testing.assert_allclose(spectra.enstrophy[0], 4.5, rtol=1e-12)


def test_spectra_are_finite_non_negative_and_shaped_for_a_real_field():
    omega = np.asarray(band_limited_initial_condition(10_000))

    spectra = radial_spectra(omega)

    assert spectra.energy.shape == (1, spectra.wavenumber.shape[0])
    assert spectra.enstrophy.shape == (1, spectra.wavenumber.shape[0])
    assert np.all(np.isfinite(spectra.energy))
    assert np.all(np.isfinite(spectra.enstrophy))
    assert np.all(spectra.energy >= 0.0)
    assert np.all(spectra.enstrophy >= 0.0)


def test_band_limited_initial_condition_has_no_enstrophy_outside_its_band():
    omega = np.asarray(band_limited_initial_condition(10_000))

    spectra = radial_spectra(omega)
    outside = spectra.enstrophy[0, :9].sum() + spectra.enstrophy[0, 34:].sum()

    assert outside < 1e-12 * spectra.enstrophy.sum()


def test_leading_axes_are_preserved_and_match_per_field_spectra():
    rng = np.random.default_rng(1)
    stack = rng.standard_normal((3, 2, COARSE_NUM_POINTS, COARSE_NUM_POINTS))

    stacked = radial_spectra(stack)

    assert stacked.energy.shape == (3, 2, stacked.wavenumber.shape[0])
    np.testing.assert_allclose(
        stacked.enstrophy[2, 1],
        radial_spectra(stack[2, 1]).enstrophy,
        rtol=1e-12,
    )


def test_domain_extent_rescales_the_energy_spectrum():
    omega = _single_mode_field(COARSE_NUM_POINTS, 4, 0)

    unit = radial_spectra(omega, domain_extent=DOMAIN_EXTENT)
    doubled = radial_spectra(omega, domain_extent=2.0 * DOMAIN_EXTENT)

    # Halving every physical wavenumber quadruples E = 0.5 |omega_hat|^2 / k^2.
    np.testing.assert_allclose(doubled.energy[4], 4.0 * unit.energy[4], rtol=1e-12)
    np.testing.assert_allclose(doubled.enstrophy[4], unit.enstrophy[4], rtol=1e-12)


def test_truncated_drops_the_mean_mode_and_the_unresolved_corners():
    rng = np.random.default_rng(2)
    omega = rng.standard_normal((COARSE_NUM_POINTS, COARSE_NUM_POINTS))

    spectra = radial_spectra(omega).truncated(coarse_grid_max_wavenumber())

    assert spectra.wavenumber[0] == 1
    assert spectra.wavenumber[-1] == COARSE_NUM_POINTS // 2
    assert spectra.energy.shape[-1] == spectra.wavenumber.shape[0]


def test_spectral_distance_is_zero_for_identical_spectra_and_positive_otherwise():
    reference = np.asarray([1.0, 0.5, 0.25])

    assert spectral_distance(reference, reference) == 0.0
    assert spectral_distance(2.0 * reference, reference) > 0.0
    np.testing.assert_allclose(
        spectral_distance(np.exp(1.0) * reference, reference),
        1.0,
        rtol=1e-6,
    )


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="square spatial grid"):
        radial_spectra(np.zeros((8, 16)))
    with pytest.raises(ValueError, match="finite"):
        radial_spectra(np.full((8, 8), np.nan))
    with pytest.raises(ValueError, match="domain_extent"):
        radial_spectra(np.zeros((8, 8)), domain_extent=0.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        spectral_distance(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="max_wavenumber"):
        radial_spectra(np.zeros((8, 8))).truncated(0)
    with pytest.raises(ValueError, match="wavenumber shells"):
        RadialSpectra(
            wavenumber=np.arange(3),
            energy=np.zeros(4),
            enstrophy=np.zeros(3),
        )


def test_decomposition_totals_reproduce_the_vorticity_mse():
    rng = np.random.default_rng(3)
    model = rng.standard_normal((COARSE_NUM_POINTS, COARSE_NUM_POINTS))
    reference = rng.standard_normal((COARSE_NUM_POINTS, COARSE_NUM_POINTS))

    decomposition = error_decomposition(model, reference)

    np.testing.assert_allclose(
        decomposition.total,
        np.mean((model - reference) ** 2),
        rtol=1e-10,
    )


def test_a_pure_amplitude_change_produces_no_phase_error():
    omega = _single_mode_field(COARSE_NUM_POINTS, 6, 0)

    decomposition = error_decomposition(0.5 * omega, omega)

    assert decomposition.phase.sum() < 1e-20
    np.testing.assert_allclose(
        decomposition.amplitude.sum(),
        np.mean((0.5 * omega - omega) ** 2),
        rtol=1e-10,
    )
    assert decomposition.phase_fraction < 1e-12


def test_a_pure_shift_produces_no_amplitude_error():
    # Translating a field leaves every mode's magnitude unchanged and moves
    # only its phase, so the whole error must land in the phase term.
    axis = 2.0 * np.pi * np.arange(COARSE_NUM_POINTS) / COARSE_NUM_POINTS
    x, y = np.meshgrid(axis, axis, indexing="ij")
    omega = np.cos(4 * x + 2 * y)
    shifted = np.roll(omega, 5, axis=0)

    decomposition = error_decomposition(shifted, omega)

    assert decomposition.amplitude.sum() < 1e-20
    np.testing.assert_allclose(
        decomposition.phase.sum(),
        np.mean((shifted - omega) ** 2),
        rtol=1e-10,
    )
    np.testing.assert_allclose(decomposition.phase_fraction, 1.0, rtol=1e-10)


def test_both_error_terms_are_non_negative_and_identical_fields_give_zero():
    rng = np.random.default_rng(4)
    omega = rng.standard_normal((2, COARSE_NUM_POINTS, COARSE_NUM_POINTS))

    identical = error_decomposition(omega, omega)
    assert identical.total.shape == (2,)
    # Not exactly zero: the two terms are formed by different float paths, so
    # cancellation leaves round-off at the scale of the field's own magnitude.
    np.testing.assert_allclose(identical.total, 0.0, atol=1e-14)

    perturbed = error_decomposition(omega + 0.1, omega)
    assert np.all(perturbed.amplitude >= 0.0)
    assert np.all(perturbed.phase >= 0.0)


def test_decomposition_preserves_leading_axes_and_truncates():
    rng = np.random.default_rng(5)
    model = rng.standard_normal((3, COARSE_NUM_POINTS, COARSE_NUM_POINTS))
    reference = rng.standard_normal((3, COARSE_NUM_POINTS, COARSE_NUM_POINTS))

    decomposition = error_decomposition(model, reference)
    assert decomposition.amplitude.shape[0] == 3

    truncated = decomposition.truncated(coarse_grid_max_wavenumber())
    assert truncated.wavenumber[0] == 1
    assert truncated.wavenumber[-1] == COARSE_NUM_POINTS // 2
    np.testing.assert_allclose(
        decomposition.amplitude[1],
        error_decomposition(model[1], reference[1]).amplitude,
        rtol=1e-10,
    )


def test_decomposition_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape mismatch"):
        error_decomposition(np.zeros((8, 8)), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="square spatial grid"):
        error_decomposition(np.zeros((4, 8)), np.zeros((4, 8)))
    with pytest.raises(ValueError, match="finite"):
        error_decomposition(np.full((8, 8), np.nan), np.zeros((8, 8)))
    with pytest.raises(ValueError, match="max_wavenumber"):
        error_decomposition(np.zeros((8, 8)), np.zeros((8, 8))).truncated(0)
