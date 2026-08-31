"""Timestep propagation through the composed Tesseracts and the demo update."""

import math

import jax
import jax.numpy as jnp
import optax
import pytest

from tesseract_hybrid_closure.closure import initial_parameters, parameter_count
from tesseract_hybrid_closure.components import operator_split_step
from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
from tesseract_hybrid_closure.data import generate_reference_trajectory
from tesseract_hybrid_closure.engine import smoke_initial_condition
from tesseract_hybrid_closure.losses import vorticity_mse
from tesseract_hybrid_closure.solver import CoarseVorticityStepper
from tesseract_hybrid_closure.tesseract_components import (
    composed_tesseract_rollout,
    composed_tesseract_step,
    local_tesseract_clients,
)
from tesseract_hybrid_closure.tesseract_demo import (
    DEMO_DT,
    DEMO_LEARNING_RATE,
    DEMO_OBJECTIVE,
    DEMO_SEED,
    DEMO_UNROLL,
    run_optimiser_demo,
)
from tesseract_hybrid_closure.tesseract_instrumentation import (
    composition_invariant_violations,
)

DEMO_VORTICITY_AMPLITUDE = 20.0


def _clients_and_inputs():
    solver, closure = local_tesseract_clients()
    omega = smoke_initial_condition(SolverConfig())
    params = jnp.asarray(initial_parameters())
    return solver, closure, omega, params


def _demo_reference():
    dns_config = DNSConfig(
        dt=DEMO_DT,
        vorticity_amplitude=DEMO_VORTICITY_AMPLITUDE,
    )
    return generate_reference_trajectory(
        DEMO_SEED,
        DEMO_UNROLL,
        split="train",
        config=dns_config,
    )


def _two_step_objective(solver, closure, candidate, initial_state, target_states):
    rollout = composed_tesseract_rollout(
        solver,
        closure,
        candidate,
        initial_state,
        num_steps=DEMO_UNROLL,
        dt=DEMO_DT,
    )
    return vorticity_mse(rollout, target_states)


def test_composed_tesseract_step_propagates_timestep():
    solver, closure, omega, params = _clients_and_inputs()
    dt = 2.0e-3

    composed = composed_tesseract_step(solver, closure, params, omega, dt=dt)
    direct = operator_split_step(
        CoarseVorticityStepper(SolverConfig(dt=dt)),
        params,
        omega,
    )
    assert jnp.allclose(composed, direct, atol=1.0e-6)

    default_composed = composed_tesseract_step(solver, closure, params, omega)
    default_direct = operator_split_step(CoarseVorticityStepper(), params, omega)
    assert jnp.allclose(default_composed, default_direct, atol=1.0e-6)

    # The default dt unchanged and the explicit dt differ numerically.
    assert not jnp.allclose(default_composed, composed)


def test_composed_tesseract_step_rejects_non_positive_timestep():
    solver, closure, omega, params = _clients_and_inputs()
    with pytest.raises(ValueError, match="finite and positive"):
        composed_tesseract_step(solver, closure, params, omega, dt=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        composed_tesseract_step(solver, closure, params, omega, dt=-1.0)


def test_optimiser_demo_rejects_invalid_max_updates():
    with pytest.raises(ValueError, match="max_updates must be positive"):
        run_optimiser_demo(max_updates=0)
    with pytest.raises(ValueError, match="max_updates must be positive"):
        run_optimiser_demo(max_updates=-2)
    with pytest.raises(TypeError, match="max_updates must be an integer"):
        run_optimiser_demo(max_updates=1.5)
    with pytest.raises(TypeError, match="max_updates must be an integer"):
        run_optimiser_demo(max_updates=True)


def test_optimiser_demo_applies_one_accepted_adam_update():
    result = run_optimiser_demo()

    assert result.seed == DEMO_SEED
    assert result.dt == DEMO_DT
    assert result.learning_rate == DEMO_LEARNING_RATE
    assert result.gradient_size == parameter_count()
    assert result.gradient_finite
    assert result.gradient_norm > 0.0
    assert result.accepted_updates == 1
    assert result.loss_improved
    assert result.loss_after < result.loss_before
    assert result.loss_after >= 0.0


@pytest.mark.parametrize("use_images", [False, True])
def test_optimiser_demo_uses_two_step_rolled_out_objective(
    use_images: bool,
):
    if use_images:
        import shutil
        import subprocess

        if shutil.which("docker") is None:
            pytest.skip("Docker unavailable")
        try:
            images = set(
                subprocess.run(
                    ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                ).stdout.split()
            )
        except (OSError, subprocess.SubprocessError):
            pytest.skip("cannot list Docker images")
        if not {"coarse_solver:0.2.0", "scalar_closure:0.2.0"} <= images:
            pytest.skip("missing container images; run `make build-tesseracts`")

    result = run_optimiser_demo(use_images=use_images)

    assert result.unroll_steps == DEMO_UNROLL == 2
    assert result.objective == DEMO_OBJECTIVE
    # The report records the actual endpoint counts; the assertions pin the
    # evidence invariants (lower bounds and path requirements) rather than the
    # exact counts, which depend on trace pruning and evaluation count.
    assert (
        composition_invariant_violations(
            solver_apply_calls=result.solver_apply_calls,
            closure_apply_calls=result.closure_apply_calls,
            solver_vjp_calls=result.solver_vjp_calls,
            closure_vjp_calls=result.closure_vjp_calls,
            solver_vjp_input_paths=result.solver_vjp_input_paths,
            closure_vjp_input_paths=result.closure_vjp_input_paths,
            solver_vjp_min_cotangent_norm=result.solver_vjp_min_cotangent_norm,
            closure_vjp_min_cotangent_norm=result.closure_vjp_min_cotangent_norm,
        )
        == []
    )


def test_parameter_update_norm_matches_the_accepted_parameter_delta():
    result = run_optimiser_demo()

    # Independently recompute the candidate exactly as the demo does.
    reference = _demo_reference()
    target_states = reference.targets
    initial_state = reference.initial_coarse
    solver, closure = local_tesseract_clients()
    params = jnp.asarray(initial_parameters(), dtype=jnp.float32)

    def objective(candidate):
        return _two_step_objective(
            solver,
            closure,
            candidate,
            initial_state,
            target_states,
        )

    gradient = jax.grad(objective)(params)
    optimiser = optax.adam(DEMO_LEARNING_RATE)
    optimiser_state = optimiser.init(params)
    updates, _ = optimiser.update(gradient, optimiser_state, params)
    candidate = optax.apply_updates(params, updates)
    expected_delta_norm = float(jnp.linalg.norm(candidate - params))

    if result.loss_improved:
        # The composed gradient path varies by a few float32 ulps between
        # independent computations (observed relative spread ~2e-5), so pin
        # the accepted-delta semantics with a noise-tolerant comparison.
        assert result.parameter_update_norm == pytest.approx(
            expected_delta_norm, rel=1.0e-4
        )
        assert result.parameter_update_norm > 0.0
    else:
        assert result.parameter_update_norm == 0.0
        assert result.loss_after == result.loss_before


def test_demo_records_solver_transpose_sensitivity():
    result = run_optimiser_demo()

    # Zeroing the solver VJP's omega cotangent must materially change the
    # two-step parameter gradient (measured approximately 0.40 locally); a
    # value near zero would mean the solver transpose contributes nothing.
    assert 0.05 < result.solver_transpose_sensitivity < 0.95
    assert math.isfinite(result.solver_transpose_sensitivity)
