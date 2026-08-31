"""Locked trajectory and a-priori baseline losses."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .closure import apply_closure
from .components import operator_split_step
from .solver import CoarseVorticityStepper


def closure_rollout(
    stepper: CoarseVorticityStepper,
    params_flat: jax.Array,
    initial_state: jax.Array,
    num_steps: int,
) -> jax.Array:
    """Roll out the hybrid model, rematerialising each composed step."""
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")

    rematerialised_step = jax.checkpoint(
        lambda state: operator_split_step(stepper, params_flat, state)
    )

    def scan_step(state: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        next_state = rematerialised_step(state)
        return next_state, next_state

    _, trajectory = jax.lax.scan(
        scan_step,
        initial_state,
        None,
        length=int(num_steps),
    )
    return trajectory


def no_closure_rollout(
    stepper: CoarseVorticityStepper,
    initial_state: jax.Array,
    num_steps: int,
) -> jax.Array:
    """Roll out the under-resolved Exponax baseline without a closure."""
    if int(num_steps) <= 0:
        raise ValueError("num_steps must be positive")

    def scan_step(state: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        next_state = stepper(state)
        return next_state, next_state

    _, trajectory = jax.lax.scan(
        scan_step,
        initial_state,
        None,
        length=int(num_steps),
    )
    return trajectory


def vorticity_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    """Return the mean squared vorticity error over all rollout values."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    return jnp.mean((prediction - target) ** 2)


def aposteriori_loss(
    stepper: CoarseVorticityStepper,
    params_flat: jax.Array,
    initial_state: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Compute solver-in-the-loop rollout vorticity MSE."""
    prediction = closure_rollout(
        stepper,
        params_flat,
        initial_state,
        targets.shape[0],
    )
    return vorticity_mse(prediction, targets)


def apriori_tendency_target(
    stepper: CoarseVorticityStepper,
    coarse_state: jax.Array,
    filtered_dns_next: jax.Array,
) -> jax.Array:
    """Infer the one-step closure tendency from a filtered DNS transition."""
    base_next = stepper(coarse_state)
    dt = jnp.asarray(stepper.config.dt, dtype=coarse_state.dtype)
    return (filtered_dns_next - base_next) / dt


def apriori_loss(
    stepper: CoarseVorticityStepper,
    params_flat: jax.Array,
    coarse_state: jax.Array,
    filtered_dns_next: jax.Array,
) -> jax.Array:
    """Fit the instantaneous scalar tendency without solver gradients."""
    target = jax.lax.stop_gradient(
        apriori_tendency_target(stepper, coarse_state, filtered_dns_next)
    )
    prediction = apply_closure(params_flat, coarse_state)
    return vorticity_mse(prediction, target)
