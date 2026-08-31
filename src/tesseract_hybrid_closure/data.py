"""Deterministic DNS trajectories and sharp spectral coarse-graining."""

from __future__ import annotations

from dataclasses import dataclass

import exponax as ex
import jax
import jax.numpy as jnp

from .configs import DNSConfig, seed_range_for_split
from .constants import (
    COARSE_NUM_POINTS,
    INITIAL_CONDITION_MAX_MODE,
    INITIAL_CONDITION_MIN_MODE,
    NUM_SPATIAL_DIMS,
)


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Coarse initial state and filtered post-step DNS targets."""

    initial_coarse: jax.Array
    targets: jax.Array


class DNSVorticityStepper:
    """Validated 256² Exponax reference stepper."""

    def __init__(self, config: DNSConfig | None = None) -> None:
        self.config = config or DNSConfig()
        self._stepper = ex.stepper.NavierStokesVorticity(
            num_spatial_dims=NUM_SPATIAL_DIMS,
            domain_extent=self.config.domain_extent,
            num_points=self.config.num_points,
            dt=self.config.dt,
            diffusivity=self.config.diffusivity,
            order=self.config.order,
        )

    def __call__(self, omega: jax.Array) -> jax.Array:
        expected_shape = (1, self.config.num_points, self.config.num_points)
        if omega.shape != expected_shape:
            raise ValueError(
                f"omega must have shape {expected_shape}, got {omega.shape}"
            )
        if omega.dtype != jnp.float32:
            raise TypeError(f"omega must have dtype float32, got {omega.dtype}")
        return self._stepper(omega)


def band_limited_initial_condition(
    seed: int,
    config: DNSConfig | None = None,
) -> jax.Array:
    """Sample a mean-zero GRF supported on radial modes 10 ≤ |k| ≤ 32."""
    resolved = config or DNSConfig()
    key = jax.random.key(int(seed))
    noise = jax.random.normal(
        key,
        (1, resolved.num_points, resolved.num_points),
        dtype=jnp.float32,
    )
    noise_hat = jnp.fft.rfftn(noise, axes=(-2, -1))
    kx = jnp.fft.fftfreq(resolved.num_points) * resolved.num_points
    ky = jnp.fft.rfftfreq(resolved.num_points) * resolved.num_points
    radial_mode = jnp.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    band = (radial_mode >= INITIAL_CONDITION_MIN_MODE) & (
        radial_mode <= INITIAL_CONDITION_MAX_MODE
    )
    field = jnp.fft.irfftn(
        noise_hat * band[jnp.newaxis, ...],
        s=(resolved.num_points, resolved.num_points),
        axes=(-2, -1),
    ).real
    field = field - jnp.mean(field, axis=(-2, -1), keepdims=True)
    maximum = jnp.max(jnp.abs(field))
    normalised = field / maximum
    return (resolved.vorticity_amplitude * normalised).astype(jnp.float32)


def sharp_spectral_filter(
    dns_state: jax.Array,
    new_num_points: int = COARSE_NUM_POINTS,
) -> jax.Array:
    """Truncate the Fourier representation to the coarse-grid modes."""
    if dns_state.ndim != 3 or dns_state.shape[0] != 1:
        raise ValueError(f"dns_state must have shape (1, N, N), got {dns_state.shape}")
    if dns_state.shape[-2] != dns_state.shape[-1]:
        raise ValueError("dns_state must use a square spatial grid")
    if new_num_points > dns_state.shape[-1]:
        raise ValueError("sharp_spectral_filter does not upsample")
    return ex.map_between_resolutions(dns_state, new_num_points)


def generate_reference_trajectory(
    seed: int,
    num_steps: int,
    *,
    split: str,
    config: DNSConfig | None = None,
) -> ReferenceTrajectory:
    """Generate one on-demand DNS rollout from an authorised split seed."""
    allowed_seeds = seed_range_for_split(split)
    if seed not in allowed_seeds:
        raise ValueError(f"seed {seed} is outside the {split!r} split")
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")

    resolved = config or DNSConfig()
    stepper = DNSVorticityStepper(resolved)
    initial_dns = band_limited_initial_condition(seed, resolved)

    def scan_step(state: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        next_state = stepper(state)
        return next_state, sharp_spectral_filter(next_state)

    _, targets = jax.lax.scan(scan_step, initial_dns, None, length=int(num_steps))
    return ReferenceTrajectory(
        initial_coarse=sharp_spectral_filter(initial_dns),
        targets=jax.lax.stop_gradient(targets),
    )
