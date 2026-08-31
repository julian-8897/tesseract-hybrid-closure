"""Short, no-training calibration of the decaying-turbulence SGS signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np

from .configs import DNSConfig, SolverConfig
from .data import generate_reference_trajectory
from .losses import no_closure_rollout, vorticity_mse
from .solver import CoarseVorticityStepper


@dataclass(frozen=True)
class RegimeCalibrationResult:
    """Aggregated no-closure discrepancy for one amplitude/timestep pair."""

    vorticity_amplitude: float
    dt: float
    num_steps: int
    num_trajectories: int
    physical_horizon: float
    mean_no_closure_mse: float
    std_no_closure_mse: float
    maximum_vorticity_growth: float
    finite: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def calibrate_regimes(
    *,
    amplitudes: tuple[float, ...],
    timesteps: tuple[float, ...],
    seeds: tuple[int, ...],
    num_steps: int = 30,
) -> tuple[RegimeCalibrationResult, ...]:
    """Measure the 256²→64² closure gap without fitting a model."""
    if not amplitudes or not timesteps or not seeds:
        raise ValueError("amplitudes, timesteps, and seeds must be non-empty")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    results = []
    for amplitude in amplitudes:
        for dt in timesteps:
            dns_config = DNSConfig(dt=dt, vorticity_amplitude=amplitude)
            solver_config = SolverConfig(dt=dt)
            stepper = CoarseVorticityStepper(solver_config)
            losses = []
            growth_values = []
            finite = True
            for seed in seeds:
                reference = generate_reference_trajectory(
                    seed,
                    num_steps,
                    split="validation",
                    config=dns_config,
                )
                prediction = no_closure_rollout(
                    stepper,
                    reference.initial_coarse,
                    num_steps,
                )
                loss = vorticity_mse(prediction, reference.targets)
                initial_maximum = jnp.max(jnp.abs(reference.initial_coarse))
                trajectory_maximum = jnp.max(jnp.abs(reference.targets))
                growth = trajectory_maximum / initial_maximum
                values_are_finite = bool(
                    jnp.isfinite(loss)
                    & jnp.isfinite(growth)
                    & jnp.all(jnp.isfinite(prediction))
                    & jnp.all(jnp.isfinite(reference.targets))
                )
                finite = finite and values_are_finite
                losses.append(float(loss))
                growth_values.append(float(growth))

            loss_values = np.asarray(losses, dtype=np.float64)
            results.append(
                RegimeCalibrationResult(
                    vorticity_amplitude=float(amplitude),
                    dt=float(dt),
                    num_steps=int(num_steps),
                    num_trajectories=len(seeds),
                    physical_horizon=float(dt * num_steps),
                    mean_no_closure_mse=float(loss_values.mean()),
                    std_no_closure_mse=float(loss_values.std()),
                    maximum_vorticity_growth=float(max(growth_values)),
                    finite=finite,
                )
            )
    return tuple(results)


def strongest_finite_regime(
    results: tuple[RegimeCalibrationResult, ...],
) -> RegimeCalibrationResult:
    """Select the finite regime with the largest measured closure gap."""
    finite_results = [result for result in results if result.finite]
    if not finite_results:
        raise ValueError("no finite calibration regime was found")
    return max(finite_results, key=lambda result: result.mean_no_closure_mse)
