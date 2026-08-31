from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tesseract_hybrid_closure.checkpointing import load_training_checkpoint
from tesseract_hybrid_closure.closure import initial_parameters
from tesseract_hybrid_closure.configs import TrainingConfig
from tesseract_hybrid_closure.data import generate_reference_trajectory
from tesseract_hybrid_closure.losses import aposteriori_loss
from tesseract_hybrid_closure.solver import CoarseVorticityStepper
from tesseract_hybrid_closure.training import train_aposteriori_curriculum


@pytest.mark.rollout_smoke
def test_length_30_checkpointed_gradient_is_finite_and_nonzero():
    reference = generate_reference_trajectory(10_000, 30, split="validation")
    params = jnp.asarray(initial_parameters())
    stepper = CoarseVorticityStepper()

    loss, gradient = jax.value_and_grad(
        lambda candidate: aposteriori_loss(
            stepper,
            candidate,
            reference.initial_coarse,
            reference.targets,
        )
    )(params)

    assert bool(jnp.isfinite(loss))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


@pytest.mark.rollout_smoke
def test_one_update_per_curriculum_stage_writes_resumable_checkpoints(
    tmp_path: Path,
):
    output_dir = tmp_path / "short-curriculum"

    result = train_aposteriori_curriculum(
        TrainingConfig(updates_per_stage=(1, 1, 1)),
        output_dir,
    )

    assert tuple(stage.unroll for stage in result.stages) == (1, 5, 30)
    assert all(jnp.isfinite(stage.final_loss) for stage in result.stages)
    for stage in result.stages:
        checkpoint = load_training_checkpoint(stage.checkpoint)
        assert checkpoint["completed_unroll"] == stage.unroll
        assert jnp.isfinite(checkpoint["params_flat"]).all()
