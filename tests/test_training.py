from pathlib import Path

import jax.numpy as jnp
import pytest

from tesseract_hybrid_closure.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from tesseract_hybrid_closure.configs import (
    DNSConfig,
    SolverConfig,
    TrainingConfig,
)
from tesseract_hybrid_closure.training import evaluate_rollout_mse


def test_training_config_locks_curriculum_and_learning_rate():
    config = TrainingConfig(updates_per_stage=(1, 2, 3))

    assert config.curriculum == (1, 5, 30)
    assert config.learning_rate == 1.0e-4

    with pytest.raises(ValueError, match="curriculum"):
        TrainingConfig(updates_per_stage=(1, 2, 3), curriculum=(1, 2, 3))
    with pytest.raises(ValueError, match="learning rate"):
        TrainingConfig(updates_per_stage=(1, 2, 3), learning_rate=2.0e-4)


def test_evaluation_rejects_mismatched_dns_and_solver_timesteps():
    with pytest.raises(ValueError, match="dt must match"):
        evaluate_rollout_mse(
            None,
            split="validation",
            seeds=(10_000,),
            unroll=1,
            solver_config=SolverConfig(dt=1.0e-3),
            dns_config=DNSConfig(dt=2.0e-3),
        )


def test_checkpoint_round_trip_and_overwrite_guard(tmp_path: Path):
    path = tmp_path / "checkpoint.pkl"
    payload = {"params_flat": jnp.arange(3, dtype=jnp.float32)}

    save_training_checkpoint(path, payload)
    loaded = load_training_checkpoint(path)

    assert loaded["format"] == "hybrid-closure-aposteriori-training"
    assert loaded["version"] == 1
    assert jnp.array_equal(loaded["params_flat"], payload["params_flat"])
    with pytest.raises(FileExistsError, match="overwrite"):
        save_training_checkpoint(path, payload)
