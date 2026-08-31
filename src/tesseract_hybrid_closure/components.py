"""Composition of the JAX solver and PyTorch closure components."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .closure import apply_closure
from .solver import CoarseVorticityStepper


def operator_split_step(
    stepper: CoarseVorticityStepper,
    params_flat: jax.Array,
    omega: jax.Array,
) -> jax.Array:
    """Apply one Exponax base step followed by one explicit closure step."""
    base_state = stepper(omega)
    closure_tendency = apply_closure(params_flat, base_state)
    return (
        base_state
        + jnp.asarray(stepper.config.dt, dtype=omega.dtype) * closure_tendency
    )
