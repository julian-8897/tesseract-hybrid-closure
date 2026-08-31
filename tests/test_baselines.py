import jax.numpy as jnp

from tesseract_hybrid_closure.baselines import (
    dynamic_smagorinsky_coefficient_squared,
    dynamic_smagorinsky_tendency,
    smagorinsky_rollout,
    static_smagorinsky_tendency,
)
from tesseract_hybrid_closure.configs import SolverConfig
from tesseract_hybrid_closure.engine import smoke_initial_condition
from tesseract_hybrid_closure.solver import CoarseVorticityStepper


def test_smagorinsky_tendencies_are_finite_and_dissipative():
    config = SolverConfig()
    omega = smoke_initial_condition(config)

    for tendency_fn in (
        static_smagorinsky_tendency,
        dynamic_smagorinsky_tendency,
    ):
        tendency = tendency_fn(omega, domain_extent=config.domain_extent)
        enstrophy_production = jnp.mean(omega * tendency)

        assert tendency.shape == omega.shape
        assert tendency.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(tendency)))
        assert float(enstrophy_production) <= 1e-7


def test_dynamic_smagorinsky_coefficient_is_finite_and_nonnegative():
    config = SolverConfig()
    omega = smoke_initial_condition(config)

    coefficient_squared = dynamic_smagorinsky_coefficient_squared(
        omega,
        domain_extent=config.domain_extent,
    )

    assert coefficient_squared.shape == ()
    assert bool(jnp.isfinite(coefficient_squared))
    assert float(coefficient_squared) >= 0.0


def test_smagorinsky_rollouts_preserve_state_invariants():
    config = SolverConfig()
    omega = smoke_initial_condition(config)
    stepper = CoarseVorticityStepper(config)

    for dynamic in (False, True):
        trajectory = smagorinsky_rollout(stepper, omega, 2, dynamic=dynamic)

        assert trajectory.shape == (2, 1, 64, 64)
        assert trajectory.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(trajectory)))
