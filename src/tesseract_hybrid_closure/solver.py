"""JAX/Exponax coarse vorticity solver."""

from __future__ import annotations

import exponax as ex
import jax
import jax.numpy as jnp

from .configs import SolverConfig
from .constants import NUM_SPATIAL_DIMS


class CoarseVorticityStepper:
    """Validated wrapper around Exponax ``NavierStokesVorticity``.

    Pins the locked :class:`~.configs.SolverConfig` (ETDRK order 2) and
    validates every field it reads, so no alternative solver configuration
    can silently reach the stepper.
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()
        self._stepper = ex.stepper.NavierStokesVorticity(
            num_spatial_dims=NUM_SPATIAL_DIMS,
            domain_extent=self.config.domain_extent,
            num_points=self.config.num_points,
            dt=self.config.dt,
            diffusivity=self.config.diffusivity,
            order=self.config.order,
        )

    def __call__(self, omega: jax.Array) -> jax.Array:
        """Advance one coarse step at the configured ETDRK order."""
        expected_shape = (1, self.config.num_points, self.config.num_points)
        if omega.shape != expected_shape:
            raise ValueError(
                f"omega must have shape {expected_shape}, got {omega.shape}"
            )
        if omega.dtype != jnp.float32:
            raise TypeError(f"omega must have dtype float32, got {omega.dtype}")
        return self._stepper(omega)
