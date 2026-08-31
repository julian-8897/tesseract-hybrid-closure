import jax.numpy as jnp
import numpy as np
import pytest

from tesseract_hybrid_closure.configs import DNSConfig, seed_range_for_split
from tesseract_hybrid_closure.constants import (
    INITIAL_CONDITION_MAX_MODE,
    INITIAL_CONDITION_MIN_MODE,
)
from tesseract_hybrid_closure.data import (
    band_limited_initial_condition,
    generate_reference_trajectory,
    sharp_spectral_filter,
)


def test_seed_ranges_are_disjoint():
    train = set(seed_range_for_split("train"))
    validation = set(seed_range_for_split("validation"))
    test = set(seed_range_for_split("test"))

    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_band_limited_initial_condition_is_reproducible_and_supported():
    state = band_limited_initial_condition(0)
    repeated = band_limited_initial_condition(0)
    spectrum = np.abs(np.fft.rfftn(np.asarray(state), axes=(-2, -1)))
    n = state.shape[-1]
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.rfftfreq(n) * n
    radial_mode = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    outside_band = (radial_mode < INITIAL_CONDITION_MIN_MODE) | (
        radial_mode > INITIAL_CONDITION_MAX_MODE
    )

    assert state.shape == (1, 256, 256)
    assert state.dtype == jnp.float32
    np.testing.assert_array_equal(state, repeated)
    assert abs(float(jnp.mean(state))) < 1e-6
    assert float(jnp.max(jnp.abs(state))) == pytest.approx(1.0)
    assert np.max(spectrum[0][outside_band]) < 2e-3


def test_initial_condition_respects_configured_vorticity_amplitude():
    state = band_limited_initial_condition(
        0,
        DNSConfig(vorticity_amplitude=20.0),
    )

    assert float(jnp.max(jnp.abs(state))) == pytest.approx(20.0)


def test_sharp_spectral_filter_preserves_low_mode_and_removes_high_mode():
    config = DNSConfig()
    coordinates = jnp.linspace(
        0.0,
        config.domain_extent,
        config.num_points,
        endpoint=False,
        dtype=jnp.float32,
    )
    x, _ = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    state = (jnp.sin(4.0 * x) + 0.5 * jnp.sin(48.0 * x))[None, ...]
    filtered = sharp_spectral_filter(state)
    coarse_coordinates = jnp.linspace(
        0.0,
        config.domain_extent,
        64,
        endpoint=False,
        dtype=jnp.float32,
    )
    coarse_x, _ = jnp.meshgrid(coarse_coordinates, coarse_coordinates, indexing="ij")

    np.testing.assert_allclose(filtered[0], jnp.sin(4.0 * coarse_x), atol=2e-5)


def test_reference_trajectory_has_coarse_finite_targets():
    trajectory = generate_reference_trajectory(10_000, 1, split="validation")

    assert trajectory.initial_coarse.shape == (1, 64, 64)
    assert trajectory.targets.shape == (1, 1, 64, 64)
    assert trajectory.initial_coarse.dtype == jnp.float32
    assert trajectory.targets.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(trajectory.targets)))


def test_reference_trajectory_rejects_cross_split_seed():
    with pytest.raises(ValueError, match="outside"):
        generate_reference_trajectory(0, 1, split="validation")
