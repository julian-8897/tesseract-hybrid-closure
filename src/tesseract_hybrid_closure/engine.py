"""Single-step cross-framework gradient-path smoke execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .closure import initial_parameters, parameter_count
from .components import operator_split_step
from .configs import SolverConfig
from .solver import CoarseVorticityStepper


@dataclass(frozen=True)
class SmokeResult:
    """Machine-readable evidence for the single-step cross-framework gradient path."""

    loss: float
    gradient_norm: float
    gradient_size: int
    state_shape: tuple[int, ...]
    state_dtype: str
    finite: bool
    nonzero: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def smoke_initial_condition(config: SolverConfig) -> jax.Array:
    """Return a deterministic smooth, mean-zero 2D vorticity field."""
    coordinates = jnp.linspace(
        0.0,
        config.domain_extent,
        config.num_points,
        endpoint=False,
        dtype=jnp.float32,
    )
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    omega = jnp.sin(2.0 * x) * jnp.cos(3.0 * y)
    return omega[jnp.newaxis, ...]


def run_gradient_smoke(config: SolverConfig | None = None) -> SmokeResult:
    """Differentiate a one-step trajectory loss through Exponax and PyTorch."""
    resolved_config = config or SolverConfig()
    stepper = CoarseVorticityStepper(resolved_config)
    omega = smoke_initial_condition(resolved_config)
    params = jnp.asarray(initial_parameters(), dtype=jnp.float32)

    def trajectory_loss(params_flat: jax.Array) -> jax.Array:
        next_state = operator_split_step(stepper, params_flat, omega)
        return jnp.mean(next_state**2)

    loss, gradient = jax.value_and_grad(trajectory_loss)(params)
    gradient_array = np.asarray(gradient)
    finite = bool(np.isfinite(gradient_array).all() and np.isfinite(float(loss)))
    gradient_norm = float(np.linalg.norm(gradient_array.astype(np.float64)))
    return SmokeResult(
        loss=float(loss),
        gradient_norm=gradient_norm,
        gradient_size=int(gradient.size),
        state_shape=tuple(omega.shape),
        state_dtype=str(omega.dtype),
        finite=finite,
        nonzero=bool(gradient_norm > 0.0),
    )


def assert_smoke_passes(result: SmokeResult) -> None:
    """Raise with a useful error if the gradient-path milestone failed."""
    if result.gradient_size != parameter_count():
        raise AssertionError(
            f"gradient has {result.gradient_size} values; expected {parameter_count()}"
        )
    if not result.finite:
        raise AssertionError("loss or closure gradient contains a non-finite value")
    if not result.nonzero:
        raise AssertionError("closure gradient is identically zero")
