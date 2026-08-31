"""A-posteriori curriculum training for the decaying-turbulence system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .baselines import smagorinsky_rollout
from .checkpointing import load_training_checkpoint, save_training_checkpoint
from .closure import initial_parameters
from .configs import DNSConfig, SolverConfig, TrainingConfig, seed_range_for_split
from .data import generate_reference_trajectory
from .losses import aposteriori_loss, apriori_loss, no_closure_rollout, vorticity_mse
from .solver import CoarseVorticityStepper


@dataclass(frozen=True)
class StageResult:
    """Loss summary for one unroll stage."""

    unroll: int
    updates: int
    first_loss: float
    final_loss: float
    first_seed: int
    final_seed: int
    checkpoint: str


@dataclass(frozen=True)
class TrainingResult:
    """Completed curriculum state and summaries."""

    params_flat: jax.Array
    stages: tuple[StageResult, ...]


def _device_tree(tree):
    return jax.tree.map(lambda value: np.asarray(jax.device_get(value)), tree)


def _resolve_matched_configs(
    solver_config: SolverConfig | None,
    dns_config: DNSConfig | None,
) -> tuple[SolverConfig, DNSConfig]:
    solver = solver_config or SolverConfig()
    dns = dns_config or DNSConfig(
        domain_extent=solver.domain_extent,
        dt=solver.dt,
        diffusivity=solver.diffusivity,
        order=solver.order,
    )
    for name in ("domain_extent", "dt", "diffusivity", "order"):
        if getattr(solver, name) != getattr(dns, name):
            raise ValueError(f"solver and DNS {name} must match")
    return solver, dns


def train_aposteriori_curriculum(
    config: TrainingConfig,
    output_dir: str | Path,
    *,
    initial_params: jax.Array | None = None,
    solver_config: SolverConfig | None = None,
    dns_config: DNSConfig | None = None,
) -> TrainingResult:
    """Train over the locked 1→5→30 curriculum and checkpoint each stage."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(
            f"refusing to reuse training output directory: {destination}"
        )
    destination.mkdir(parents=True)

    resolved_solver, resolved_dns = _resolve_matched_configs(
        solver_config,
        dns_config,
    )
    stepper = CoarseVorticityStepper(resolved_solver)
    params = (
        jnp.asarray(initial_parameters(), dtype=jnp.float32)
        if initial_params is None
        else jnp.asarray(initial_params, dtype=jnp.float32)
    )
    optimiser = optax.adam(config.learning_rate)
    optimiser_state = optimiser.init(params)
    update_index = 0
    stage_results: list[StageResult] = []

    for unroll, num_updates in zip(
        config.curriculum, config.updates_per_stage, strict=True
    ):

        @jax.jit
        def loss_and_gradient(
            current_params: jax.Array,
            initial_state: jax.Array,
            targets: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return jax.value_and_grad(
                lambda candidate: aposteriori_loss(
                    stepper,
                    candidate,
                    initial_state,
                    targets,
                )
            )(current_params)

        losses: list[float] = []
        first_seed = seed_range_for_split("train")[update_index]
        for _ in range(num_updates):
            seed = seed_range_for_split("train")[update_index]
            reference = generate_reference_trajectory(
                seed,
                unroll,
                split="train",
                config=resolved_dns,
            )
            loss, gradient = loss_and_gradient(
                params,
                reference.initial_coarse,
                reference.targets,
            )
            if not bool(jnp.isfinite(loss)) or not bool(
                jnp.all(jnp.isfinite(gradient))
            ):
                raise FloatingPointError(
                    f"non-finite loss or gradient at update {update_index}, seed {seed}"
                )
            updates, optimiser_state = optimiser.update(
                gradient,
                optimiser_state,
                params,
            )
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
            update_index += 1

        final_seed = seed_range_for_split("train")[update_index - 1]
        checkpoint_path = destination / f"stage-unroll-{unroll}.pkl"
        save_training_checkpoint(
            checkpoint_path,
            {
                "params_flat": np.asarray(jax.device_get(params)),
                "optimiser_state": _device_tree(optimiser_state),
                "training_config": asdict(config),
                "solver_config": asdict(resolved_solver),
                "dns_config": asdict(resolved_dns),
                "completed_updates": update_index,
                "completed_unroll": unroll,
                "losses": losses,
            },
        )
        stage_results.append(
            StageResult(
                unroll=unroll,
                updates=num_updates,
                first_loss=losses[0],
                final_loss=losses[-1],
                first_seed=first_seed,
                final_seed=final_seed,
                checkpoint=str(checkpoint_path),
            )
        )

    return TrainingResult(params_flat=params, stages=tuple(stage_results))


def continue_aposteriori_stage(
    checkpoint_path: str | Path,
    num_updates: int,
    output_path: str | Path,
) -> StageResult:
    """Resume the stage-30 Adam state on the next unused training seeds."""
    if int(num_updates) <= 0:
        raise ValueError("num_updates must be positive")
    checkpoint = load_training_checkpoint(checkpoint_path)
    unroll = int(checkpoint.get("completed_unroll", -1))
    if unroll != 30:
        raise ValueError("only the completed unroll-30 stage can be continued")
    completed_updates = int(checkpoint["completed_updates"])
    if completed_updates + num_updates > len(seed_range_for_split("train")):
        raise ValueError("continued updates exceed the training seed range")

    solver_config = SolverConfig(**checkpoint["solver_config"])
    dns_config = DNSConfig(**checkpoint["dns_config"])
    stepper = CoarseVorticityStepper(solver_config)
    params = jnp.asarray(checkpoint["params_flat"], dtype=jnp.float32)
    optimiser = optax.adam(float(checkpoint["training_config"]["learning_rate"]))
    optimiser_state = jax.tree.map(jnp.asarray, checkpoint["optimiser_state"])

    @jax.jit
    def loss_and_gradient(
        current_params: jax.Array,
        initial_state: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return jax.value_and_grad(
            lambda candidate: aposteriori_loss(
                stepper,
                candidate,
                initial_state,
                targets,
            )
        )(current_params)

    losses: list[float] = []
    first_seed = seed_range_for_split("train")[completed_updates]
    for update_index in range(completed_updates, completed_updates + num_updates):
        seed = seed_range_for_split("train")[update_index]
        reference = generate_reference_trajectory(
            seed,
            unroll,
            split="train",
            config=dns_config,
        )
        loss, gradient = loss_and_gradient(
            params,
            reference.initial_coarse,
            reference.targets,
        )
        if not bool(jnp.isfinite(loss)) or not bool(jnp.all(jnp.isfinite(gradient))):
            raise FloatingPointError(
                f"non-finite loss or gradient at update {update_index}, seed {seed}"
            )
        updates, optimiser_state = optimiser.update(
            gradient,
            optimiser_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))

    total_updates = completed_updates + num_updates
    final_seed = seed_range_for_split("train")[total_updates - 1]
    previous_losses = list(checkpoint.get("losses", ()))
    destination = save_training_checkpoint(
        output_path,
        {
            "params_flat": np.asarray(jax.device_get(params)),
            "optimiser_state": _device_tree(optimiser_state),
            "training_config": checkpoint["training_config"],
            "solver_config": checkpoint["solver_config"],
            "dns_config": checkpoint["dns_config"],
            "completed_updates": total_updates,
            "completed_unroll": unroll,
            "losses": previous_losses + losses,
            "parent_checkpoint": str(checkpoint_path),
        },
    )
    return StageResult(
        unroll=unroll,
        updates=num_updates,
        first_loss=losses[0],
        final_loss=losses[-1],
        first_seed=first_seed,
        final_seed=final_seed,
        checkpoint=str(destination),
    )


def train_apriori_baseline(
    num_updates: int,
    *,
    initial_params: jax.Array | None = None,
    solver_config: SolverConfig | None = None,
    dns_config: DNSConfig | None = None,
) -> tuple[jax.Array, tuple[float, ...]]:
    """Train the instantaneous-tendency baseline on disjoint training seeds."""
    if int(num_updates) <= 0 or num_updates > len(seed_range_for_split("train")):
        raise ValueError("num_updates must fit within the training seed range")
    resolved_solver, resolved_dns = _resolve_matched_configs(
        solver_config,
        dns_config,
    )
    stepper = CoarseVorticityStepper(resolved_solver)
    params = (
        jnp.asarray(initial_parameters(), dtype=jnp.float32)
        if initial_params is None
        else jnp.asarray(initial_params, dtype=jnp.float32)
    )
    optimiser = optax.adam(1.0e-4)
    optimiser_state = optimiser.init(params)

    @jax.jit
    def loss_and_gradient(
        current_params: jax.Array,
        coarse_state: jax.Array,
        filtered_dns_next: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return jax.value_and_grad(
            lambda candidate: apriori_loss(
                stepper,
                candidate,
                coarse_state,
                filtered_dns_next,
            )
        )(current_params)

    losses: list[float] = []
    for seed in seed_range_for_split("train")[:num_updates]:
        reference = generate_reference_trajectory(
            seed,
            1,
            split="train",
            config=resolved_dns,
        )
        loss, gradient = loss_and_gradient(
            params,
            reference.initial_coarse,
            reference.targets[0],
        )
        updates, optimiser_state = optimiser.update(
            gradient,
            optimiser_state,
            params,
        )
        params = optax.apply_updates(params, updates)
        losses.append(float(loss))
    return params, tuple(losses)


def evaluate_rollout_mse(
    params_flat: jax.Array | None,
    *,
    split: str,
    seeds: tuple[int, ...],
    unroll: int,
    baseline: str = "none",
    solver_config: SolverConfig | None = None,
    dns_config: DNSConfig | None = None,
) -> dict[str, float | int]:
    """Evaluate a learned closure or named baseline on authorised seeds."""
    allowed = seed_range_for_split(split)
    if not seeds or any(seed not in allowed for seed in seeds):
        raise ValueError(f"all evaluation seeds must belong to the {split!r} split")
    if params_flat is not None and baseline != "none":
        raise ValueError("baseline must be 'none' when evaluating learned parameters")
    if baseline not in {"none", "static-smagorinsky", "dynamic-smagorinsky"}:
        raise ValueError(f"unknown baseline: {baseline!r}")
    resolved_solver, resolved_dns = _resolve_matched_configs(
        solver_config,
        dns_config,
    )
    stepper = CoarseVorticityStepper(resolved_solver)
    losses = []
    for seed in seeds:
        reference = generate_reference_trajectory(
            seed,
            unroll,
            split=split,
            config=resolved_dns,
        )
        if params_flat is None:
            if baseline == "none":
                prediction = no_closure_rollout(
                    stepper,
                    reference.initial_coarse,
                    unroll,
                )
            else:
                prediction = smagorinsky_rollout(
                    stepper,
                    reference.initial_coarse,
                    unroll,
                    dynamic=baseline == "dynamic-smagorinsky",
                )
            loss = vorticity_mse(prediction, reference.targets)
        else:
            loss = aposteriori_loss(
                stepper,
                jnp.asarray(params_flat, dtype=jnp.float32),
                reference.initial_coarse,
                reference.targets,
            )
        losses.append(float(loss))
    values = np.asarray(losses, dtype=np.float64)
    return {
        "mean_vorticity_mse": float(values.mean()),
        "std_vorticity_mse": float(values.std()),
        "num_trajectories": int(values.size),
    }
