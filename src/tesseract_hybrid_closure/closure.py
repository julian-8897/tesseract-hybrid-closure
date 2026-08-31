"""JAX reverse-mode bridge for the PyTorch scalar SGS closure."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .torch_closure import (
    ScalarTendencyCNN,
    initial_parameters,
    parameter_count,
    torch_forward,
    torch_vjp,
)


def _closure_callback(params_flat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return torch_forward(params_flat, omega)


def _closure_vjp_callback(
    params_flat: np.ndarray, omega: np.ndarray, cotangent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return torch_vjp(params_flat, omega, cotangent)


@jax.custom_vjp
def apply_closure(params_flat: jax.Array, omega: jax.Array) -> jax.Array:
    """Apply PyTorch through a callback carrying an explicit reverse-mode rule."""
    output_spec = jax.ShapeDtypeStruct(omega.shape, jnp.float32)
    return jax.pure_callback(
        _closure_callback,
        output_spec,
        params_flat,
        omega,
        vmap_method="sequential",
    )


def _apply_closure_fwd(
    params_flat: jax.Array, omega: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    tendency = apply_closure(params_flat, omega)
    return tendency, (params_flat, omega)


def _apply_closure_bwd(
    residuals: tuple[jax.Array, jax.Array], cotangent: jax.Array
) -> tuple[jax.Array, jax.Array]:
    params_flat, omega = residuals
    result_spec = (
        jax.ShapeDtypeStruct(params_flat.shape, jnp.float32),
        jax.ShapeDtypeStruct(omega.shape, jnp.float32),
    )
    return jax.pure_callback(
        _closure_vjp_callback,
        result_spec,
        params_flat,
        omega,
        cotangent,
        vmap_method="sequential",
    )


apply_closure.defvjp(_apply_closure_fwd, _apply_closure_bwd)

__all__ = [
    "ScalarTendencyCNN",
    "apply_closure",
    "initial_parameters",
    "parameter_count",
    "torch_forward",
    "torch_vjp",
]
