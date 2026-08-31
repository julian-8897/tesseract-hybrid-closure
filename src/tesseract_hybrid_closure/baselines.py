"""No-closure and Smagorinsky baselines for 2D vorticity LES."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .constants import (
    SMAGORINSKY_TEST_FILTER_RATIO,
    STATIC_SMAGORINSKY_COEFFICIENT,
)
from .solver import CoarseVorticityStepper


def _spectral_fields(
    omega: jax.Array,
    domain_extent: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    n = omega.shape[-1]
    dx = domain_extent / n
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
    ky = 2.0 * jnp.pi * jnp.fft.rfftfreq(n, d=dx)
    kx_grid = kx[:, None]
    ky_grid = ky[None, :]
    k_squared = kx_grid**2 + ky_grid**2
    omega_hat = jnp.fft.rfftn(omega, axes=(-2, -1))
    inverse_laplacian = jnp.where(k_squared == 0.0, 0.0, 1.0 / k_squared)
    psi_hat = omega_hat * inverse_laplacian[jnp.newaxis, ...]
    return omega_hat, psi_hat, kx_grid, ky_grid, k_squared


def _to_physical(field_hat: jax.Array, n: int) -> jax.Array:
    return jnp.fft.irfftn(field_hat, s=(n, n), axes=(-2, -1)).real


def _strain_magnitude(
    psi_hat: jax.Array,
    kx: jax.Array,
    ky: jax.Array,
    n: int,
) -> jax.Array:
    psi_xx = _to_physical(-(kx**2)[None, ...] * psi_hat, n)
    psi_yy = _to_physical(-(ky**2)[None, ...] * psi_hat, n)
    psi_xy = _to_physical(-(kx * ky)[None, ...] * psi_hat, n)
    strain_11 = psi_xy
    strain_12 = 0.5 * (psi_yy - psi_xx)
    return 2.0 * jnp.sqrt(strain_11**2 + strain_12**2)


def _jacobian(
    omega_hat: jax.Array,
    psi_hat: jax.Array,
    kx: jax.Array,
    ky: jax.Array,
    n: int,
) -> jax.Array:
    omega_x = _to_physical(1j * kx[None, ...] * omega_hat, n)
    omega_y = _to_physical(1j * ky[None, ...] * omega_hat, n)
    psi_x = _to_physical(1j * kx[None, ...] * psi_hat, n)
    psi_y = _to_physical(1j * ky[None, ...] * psi_hat, n)
    return omega_x * psi_y - omega_y * psi_x


def _test_filter_mask(n: int) -> jax.Array:
    kx_modes = jnp.fft.fftfreq(n) * n
    ky_modes = jnp.fft.rfftfreq(n) * n
    cutoff = n // (2 * SMAGORINSKY_TEST_FILTER_RATIO)
    return (jnp.abs(kx_modes[:, None]) <= cutoff) & (
        jnp.abs(ky_modes[None, :]) <= cutoff
    )


def _test_filter(field: jax.Array) -> jax.Array:
    n = field.shape[-1]
    field_hat = jnp.fft.rfftn(field, axes=(-2, -1))
    filtered_hat = field_hat * _test_filter_mask(n)[None, ...]
    return _to_physical(filtered_hat, n)


def static_smagorinsky_tendency(
    omega: jax.Array,
    *,
    domain_extent: float,
    coefficient: float = STATIC_SMAGORINSKY_COEFFICIENT,
) -> jax.Array:
    """Return the globally averaged static Smagorinsky vorticity tendency."""
    n = omega.shape[-1]
    omega_hat, psi_hat, kx, ky, k_squared = _spectral_fields(omega, domain_extent)
    strain = _strain_magnitude(psi_hat, kx, ky, n)
    delta = domain_extent / n
    eddy_viscosity = (coefficient * delta) ** 2 * jnp.sqrt(jnp.mean(strain**2))
    laplacian_omega = _to_physical(-k_squared[None, ...] * omega_hat, n)
    return eddy_viscosity * laplacian_omega


def dynamic_smagorinsky_coefficient_squared(
    omega: jax.Array,
    *,
    domain_extent: float,
) -> jax.Array:
    """Compute a clipped Germano coefficient Cₛ² using a 2× test filter."""
    n = omega.shape[-1]
    omega_hat, psi_hat, kx, ky, k_squared = _spectral_fields(omega, domain_extent)
    strain = _strain_magnitude(psi_hat, kx, ky, n)
    laplacian_omega = _to_physical(-k_squared[None, ...] * omega_hat, n)

    mask = _test_filter_mask(n)[None, ...]
    omega_filtered_hat = omega_hat * mask
    psi_filtered_hat = psi_hat * mask
    jacobian = _jacobian(omega_hat, psi_hat, kx, ky, n)
    jacobian_filtered = _test_filter(jacobian)
    filtered_jacobian = _jacobian(
        omega_filtered_hat,
        psi_filtered_hat,
        kx,
        ky,
        n,
    )
    germano_residual = jacobian_filtered - filtered_jacobian

    delta = domain_extent / n
    test_delta = SMAGORINSKY_TEST_FILTER_RATIO * delta
    filtered_laplacian = _to_physical(
        -k_squared[None, ...] * omega_filtered_hat,
        n,
    )
    model_residual = delta**2 * _test_filter(strain * laplacian_omega) - (
        test_delta**2 * _test_filter(strain) * filtered_laplacian
    )
    numerator = jnp.mean(jnp.maximum(germano_residual * model_residual, 0.0))
    denominator = jnp.mean(model_residual**2)
    return jnp.where(denominator > 0.0, numerator / denominator, 0.0)


def dynamic_smagorinsky_tendency(
    omega: jax.Array,
    *,
    domain_extent: float,
) -> jax.Array:
    """Return the globally averaged dynamic Smagorinsky vorticity tendency."""
    n = omega.shape[-1]
    omega_hat, psi_hat, kx, ky, k_squared = _spectral_fields(omega, domain_extent)
    strain = _strain_magnitude(psi_hat, kx, ky, n)
    coefficient_squared = dynamic_smagorinsky_coefficient_squared(
        omega,
        domain_extent=domain_extent,
    )
    delta = domain_extent / n
    eddy_viscosity = coefficient_squared * delta**2 * jnp.sqrt(jnp.mean(strain**2))
    laplacian_omega = _to_physical(-k_squared[None, ...] * omega_hat, n)
    return eddy_viscosity * laplacian_omega


def smagorinsky_step(
    stepper: CoarseVorticityStepper,
    omega: jax.Array,
    tendency: Callable[..., jax.Array],
) -> jax.Array:
    """Apply the same operator split used by the learned closure."""
    base_state = stepper(omega)
    correction = tendency(base_state, domain_extent=stepper.config.domain_extent)
    return base_state + jnp.asarray(stepper.config.dt, omega.dtype) * correction


def smagorinsky_rollout(
    stepper: CoarseVorticityStepper,
    initial_state: jax.Array,
    num_steps: int,
    *,
    dynamic: bool,
) -> jax.Array:
    """Roll out either dynamic or static Smagorinsky."""
    tendency = dynamic_smagorinsky_tendency if dynamic else static_smagorinsky_tendency

    def scan_step(state: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        next_state = smagorinsky_step(stepper, state, tendency)
        return next_state, next_state

    _, trajectory = jax.lax.scan(scan_step, initial_state, None, length=int(num_steps))
    return trajectory
