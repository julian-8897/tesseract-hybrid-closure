# Copyright 2026 Julian Chan
# SPDX-License-Identifier: Apache-2.0

"""Exponax ETDRK2 coarse-vorticity solver Tesseract API."""

from typing import Any

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

from tesseract_hybrid_closure.configs import SolverConfig
from tesseract_hybrid_closure.constants import COARSE_DT
from tesseract_hybrid_closure.solver import CoarseVorticityStepper

_STEPPERS: dict[float, CoarseVorticityStepper] = {}


def _stepper_for(dt: float) -> CoarseVorticityStepper:
    """Return a cached coarse stepper for one non-differentiable timestep."""
    resolved = float(dt)
    if resolved not in _STEPPERS:
        # SolverConfig raises on non-finite or non-positive timesteps.
        _STEPPERS[resolved] = CoarseVorticityStepper(SolverConfig(dt=resolved))
    return _STEPPERS[resolved]


class InputSchema(BaseModel):
    """Input schema for one coarse solver step."""

    omega: Differentiable[Array[(1, 64, 64), Float32]] = Field(
        description="Coarse vorticity with channel-first shape (1, 64, 64)"
    )
    dt: float = Field(
        default=COARSE_DT,
        gt=0.0,
        description="Non-differentiable solver timestep",
    )


class OutputSchema(BaseModel):
    """Output schema for one coarse solver step."""

    omega_next: Differentiable[Array[(1, 64, 64), Float32]] = Field(
        description="Vorticity after one Exponax ETDRK2 step"
    )


@jax.jit(static_argnums=(1,))
def apply_jit(omega: jax.Array, dt: float) -> dict[str, jax.Array]:
    omega = jnp.asarray(omega, dtype=jnp.float32)
    return {"omega_next": _stepper_for(dt)(omega)}


def apply(inputs: InputSchema) -> OutputSchema:
    """Advance the vorticity field by one coarse step."""
    return apply_jit(inputs.omega, inputs.dt)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Compute a reverse-mode product through the Exponax step."""
    if vjp_inputs != {"omega"} or vjp_outputs != {"omega_next"}:
        raise NotImplementedError(
            "The coarse solver VJP supports omega_next with respect to omega only"
        )
    omega = jnp.asarray(inputs.omega, dtype=jnp.float32)
    _, pullback = jax.vjp(lambda value: _stepper_for(inputs.dt)(value), omega)
    (omega_gradient,) = pullback(
        jnp.asarray(cotangent_vector["omega_next"], dtype=jnp.float32)
    )
    return {"omega": omega_gradient}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Compute a forward-mode product through the Exponax step."""
    if jvp_inputs != {"omega"} or jvp_outputs != {"omega_next"}:
        raise NotImplementedError(
            "The coarse solver JVP supports omega_next with respect to omega only"
        )
    omega = jnp.asarray(inputs.omega, dtype=jnp.float32)
    _, tangent = jax.jvp(
        lambda value: _stepper_for(inputs.dt)(value),
        (omega,),
        (jnp.asarray(tangent_vector["omega"], dtype=jnp.float32),),
    )
    return {"omega_next": tangent}


def abstract_eval(abstract_inputs):
    """Return the fixed output shape and dtype after validating the timestep."""
    dt = getattr(abstract_inputs, "dt", COARSE_DT)
    dt_value = dt["value"] if isinstance(dt, dict) else dt
    # SolverConfig raises on non-finite or non-positive timesteps.
    _stepper_for(dt_value)
    return {"omega_next": {"shape": (1, 64, 64), "dtype": "float32"}}
