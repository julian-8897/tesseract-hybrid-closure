"""Composition through the public Tesseract and ``tesseract-jax`` APIs."""

import jax
import jax.numpy as jnp
import pytest

from tesseract_hybrid_closure.closure import initial_parameters
from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
from tesseract_hybrid_closure.data import generate_reference_trajectory
from tesseract_hybrid_closure.engine import smoke_initial_condition
from tesseract_hybrid_closure.losses import vorticity_mse
from tesseract_hybrid_closure.tesseract_components import (
    composed_tesseract_rollout,
    composed_tesseract_step,
    local_tesseract_clients,
)
from tesseract_hybrid_closure.tesseract_demo import (
    DEMO_DT,
    DEMO_SEED,
    DEMO_UNROLL,
)
from tesseract_hybrid_closure.tesseract_instrumentation import (
    InstrumentedTesseract,
    composition_invariant_violations,
    solver_omega_zeroing_vjp_override,
)

# Locked demo regime: train-seed reference at |omega|max=20, dt=0.002.
_DEMO_VORTICITY_AMPLITUDE = 20.0


def _two_step_objective(
    solver,
    closure,
    candidate,
    initial_state,
    target_states,
):
    rollout = composed_tesseract_rollout(
        solver,
        closure,
        candidate,
        initial_state,
        num_steps=DEMO_UNROLL,
        dt=DEMO_DT,
    )
    return vorticity_mse(rollout, target_states)


@pytest.fixture(scope="module")
def demo_inputs():
    """Locked demo inputs: float32 params, coarse start state and two DNS targets."""
    dns_config = DNSConfig(
        dt=DEMO_DT,
        vorticity_amplitude=_DEMO_VORTICITY_AMPLITUDE,
    )
    reference = generate_reference_trajectory(
        DEMO_SEED,
        DEMO_UNROLL,
        split="train",
        config=dns_config,
    )
    params = jnp.asarray(initial_parameters(), dtype=jnp.float32)
    return params, reference.initial_coarse, reference.targets


def test_tesseract_jax_composition_has_end_to_end_parameter_gradient():
    solver, closure = local_tesseract_clients()
    omega = smoke_initial_condition(SolverConfig())
    params = jnp.asarray(initial_parameters())

    def loss(candidate):
        next_state = composed_tesseract_step(
            solver,
            closure,
            candidate,
            omega,
        )
        return jnp.mean(next_state**2)

    value, gradient = jax.value_and_grad(loss)(params)

    assert bool(jnp.isfinite(value))
    assert gradient.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_instrumentation_is_transparent_to_the_two_step_gradient(demo_inputs):
    """Wrapping the clients must not change the gradient, only record it.

    The recorded calls for one two-step gradient evaluation are the actual
    trace structure: two applies per endpoint, one solver VJP (the first
    solver step is pruned because the constant initial-state cotangent is
    discarded) and two closure VJPs, the second requesting only ``params_flat``
    because its base-state cotangent is routed directly into the solver VJP
    that then carries it back through step one.
    """
    params, initial_state, target_states = demo_inputs

    solver, closure = local_tesseract_clients()
    g_plain = jax.grad(
        lambda p: _two_step_objective(solver, closure, p, initial_state, target_states)
    )(params)

    instrumented_solver = InstrumentedTesseract(local_tesseract_clients()[0])
    instrumented_closure = InstrumentedTesseract(local_tesseract_clients()[1])
    g_instrumented = jax.grad(
        lambda p: _two_step_objective(
            instrumented_solver,
            instrumented_closure,
            p,
            initial_state,
            target_states,
        )
    )(params)

    assert g_plain.shape == g_instrumented.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(g_plain)))
    assert bool(jnp.all(jnp.isfinite(g_instrumented)))
    assert float(jnp.linalg.norm(g_plain)) > 0.0
    # Observed bitwise identical in-process (max abs diff 0.0); the tolerance
    # only accommodates genuine float32 reduction nondeterminism.
    assert jnp.allclose(g_plain, g_instrumented, rtol=1.0e-5, atol=1.0e-8)

    assert instrumented_solver.apply_calls == 2
    assert instrumented_closure.apply_calls == 2
    assert instrumented_solver.vjp_calls == 1
    assert instrumented_closure.vjp_calls == 2
    assert instrumented_solver.vjp_input_paths == [["omega"]]
    assert instrumented_closure.vjp_input_paths == [
        ["omega", "params_flat"],
        ["params_flat"],
    ]
    assert (
        composition_invariant_violations(
            solver_apply_calls=instrumented_solver.apply_calls,
            closure_apply_calls=instrumented_closure.apply_calls,
            solver_vjp_calls=instrumented_solver.vjp_calls,
            closure_vjp_calls=instrumented_closure.vjp_calls,
            solver_vjp_input_paths=instrumented_solver.vjp_input_paths,
            closure_vjp_input_paths=instrumented_closure.vjp_input_paths,
            solver_vjp_min_cotangent_norm=(
                min(instrumented_solver.vjp_cotangent_norms)
                if instrumented_solver.vjp_cotangent_norms
                else 0.0
            ),
            closure_vjp_min_cotangent_norm=(
                min(instrumented_closure.vjp_cotangent_norms)
                if instrumented_closure.vjp_cotangent_norms
                else 0.0
            ),
        )
        == []
    )


def test_solver_vjp_transpose_materially_changes_parameter_gradient(demo_inputs):
    """Zeroing the solver omega cotangent must alter the parameter gradient.

    The two-step rollout gradient depends on the second solver step's VJP:
    base state 2 depends on parameters through ``solver(step1)``, so the
    solver transpose carries loss cotangents back into the first closure
    correction. If the solver VJP merely executed without contributing, the
    zeroed gradient would equal the true gradient. Measured locally,
    ``||g_true - g_zero|| / ||g_true|| == 0.3998``, so the solver transpose
    carries roughly 40% of the parameter gradient.
    """
    params, initial_state, target_states = demo_inputs

    solver, closure = local_tesseract_clients()
    g_true = jax.grad(
        lambda p: _two_step_objective(solver, closure, p, initial_state, target_states)
    )(params)

    solver_client, closure_client = local_tesseract_clients()
    zeroed_solver = InstrumentedTesseract(
        solver_client,
        vjp_override=solver_omega_zeroing_vjp_override(solver_client),
    )
    instrumented_closure = InstrumentedTesseract(closure_client)
    g_zero = jax.grad(
        lambda p: _two_step_objective(
            zeroed_solver,
            instrumented_closure,
            p,
            initial_state,
            target_states,
        )
    )(params)

    assert g_true.shape == g_zero.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(g_true)))
    assert bool(jnp.all(jnp.isfinite(g_zero)))
    assert float(jnp.linalg.norm(g_true)) > 0.0
    # The zeroed run must still exercise the full composition with the same
    # VJP path structure, so the difference below cannot be an artefact of a
    # broken trace.
    assert (
        composition_invariant_violations(
            solver_apply_calls=zeroed_solver.apply_calls,
            closure_apply_calls=instrumented_closure.apply_calls,
            solver_vjp_calls=zeroed_solver.vjp_calls,
            closure_vjp_calls=instrumented_closure.vjp_calls,
            solver_vjp_input_paths=zeroed_solver.vjp_input_paths,
            closure_vjp_input_paths=instrumented_closure.vjp_input_paths,
            solver_vjp_min_cotangent_norm=(
                min(zeroed_solver.vjp_cotangent_norms)
                if zeroed_solver.vjp_cotangent_norms
                else 0.0
            ),
            closure_vjp_min_cotangent_norm=(
                min(instrumented_closure.vjp_cotangent_norms)
                if instrumented_closure.vjp_cotangent_norms
                else 0.0
            ),
        )
        == []
    )

    relative_difference = float(
        jnp.linalg.norm(g_true - g_zero) / jnp.linalg.norm(g_true)
    )
    # Bound of 0.04: 10x below the measured 0.3998, and about three orders of
    # magnitude above the ~2e-5 float32 recomputation spread observed between
    # independent executions in the demo tests, so it cannot be triggered by
    # numerical noise while still proving the solver VJP contributes.
    assert relative_difference >= 0.04
