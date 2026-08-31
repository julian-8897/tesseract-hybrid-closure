import jax
import jax.numpy as jnp

from tesseract_hybrid_closure.closure import initial_parameters
from tesseract_hybrid_closure.data import generate_reference_trajectory
from tesseract_hybrid_closure.losses import (
    aposteriori_loss,
    apriori_tendency_target,
    closure_rollout,
    no_closure_rollout,
)
from tesseract_hybrid_closure.solver import CoarseVorticityStepper


def test_rollouts_preserve_target_shape_dtype_and_finiteness():
    reference = generate_reference_trajectory(10_000, 2, split="validation")
    stepper = CoarseVorticityStepper()
    params = jnp.asarray(initial_parameters())

    learned = closure_rollout(stepper, params, reference.initial_coarse, 2)
    baseline = no_closure_rollout(stepper, reference.initial_coarse, 2)

    assert learned.shape == reference.targets.shape == (2, 1, 64, 64)
    assert baseline.shape == reference.targets.shape
    assert learned.dtype == baseline.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(learned)))
    assert bool(jnp.all(jnp.isfinite(baseline)))


def test_checkpointed_aposteriori_loss_has_finite_nonzero_parameter_gradient():
    reference = generate_reference_trajectory(10_000, 2, split="validation")
    stepper = CoarseVorticityStepper()
    params = jnp.asarray(initial_parameters())

    loss, gradient = jax.value_and_grad(
        lambda candidate: aposteriori_loss(
            stepper,
            candidate,
            reference.initial_coarse,
            reference.targets,
        )
    )(params)

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert gradient.shape == params.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_apriori_target_matches_one_step_residual_definition():
    reference = generate_reference_trajectory(10_000, 1, split="validation")
    stepper = CoarseVorticityStepper()

    target = apriori_tendency_target(
        stepper,
        reference.initial_coarse,
        reference.targets[0],
    )
    reconstructed = stepper(reference.initial_coarse) + stepper.config.dt * target

    assert target.shape == reference.initial_coarse.shape
    assert bool(jnp.all(jnp.isfinite(target)))
    assert bool(jnp.allclose(reconstructed, reference.targets[0], atol=1e-6))
