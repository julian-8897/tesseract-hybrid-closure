import jax
import jax.numpy as jnp
import numpy as np

from tesseract_hybrid_closure.closure import (
    apply_closure,
    initial_parameters,
    parameter_count,
)
from tesseract_hybrid_closure.configs import SolverConfig
from tesseract_hybrid_closure.engine import smoke_initial_condition
from tesseract_hybrid_closure.solver import CoarseVorticityStepper


def test_initial_parameters_are_reproducible_float32():
    first = initial_parameters()
    second = initial_parameters()

    assert first.shape == (parameter_count(),)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()


def test_exponax_step_preserves_shape_dtype_and_finiteness():
    config = SolverConfig()
    omega = smoke_initial_condition(config)

    result = CoarseVorticityStepper(config)(omega)

    assert result.shape == (1, 64, 64)
    assert result.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(result)))


def test_closure_vjp_covers_state_and_all_parameters():
    config = SolverConfig()
    omega = smoke_initial_condition(config)
    params = jnp.asarray(initial_parameters())

    output, pullback = jax.vjp(apply_closure, params, omega)
    params_gradient, omega_gradient = pullback(jnp.ones_like(output))

    assert output.shape == omega.shape
    assert output.dtype == jnp.float32
    assert params_gradient.shape == params.shape
    assert omega_gradient.shape == omega.shape
    assert bool(jnp.all(jnp.isfinite(params_gradient)))
    assert bool(jnp.all(jnp.isfinite(omega_gradient)))
