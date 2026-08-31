"""Tesseract-composed optimiser demo on a genuine train-split DNS target."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from .closure import initial_parameters
from .configs import DNSConfig
from .data import generate_reference_trajectory
from .losses import vorticity_mse
from .tesseract_components import (
    composed_tesseract_rollout,
    image_tesseract_clients,
    local_tesseract_clients,
    teardown_image_clients,
)
from .tesseract_instrumentation import (
    InstrumentedTesseract,
    composition_invariant_violations,
    solver_omega_zeroing_vjp_override,
)

DEMO_SEED = 0
DEMO_VORTICITY_AMPLITUDE = 20.0
DEMO_DT = 2.0e-3
DEMO_UNROLL = 2
DEMO_OBJECTIVE = (
    "two-step composed rollout vorticity-MSE against filtered DNS targets, "
    "mean over both rollout states"
)
DEMO_LEARNING_RATE = 1.0e-4
DEMO_MAX_UPDATES = 1


@dataclass(frozen=True)
class OptimiserDemoResult:
    """Metrics for one gradient step through the composed Tesseracts.

    All endpoint counters are measured at the client boundary of the running
    demo via ``InstrumentedTesseract``; they are never asserted by the caller.
    """

    loss_before: float
    loss_after: float
    gradient_norm: float
    gradient_size: int
    gradient_finite: bool
    parameter_update_norm: float
    loss_improved: bool
    accepted_updates: int
    seed: int
    dt: float
    learning_rate: float
    vorticity_amplitude: float
    use_images: bool
    unroll_steps: int
    objective: str
    solver_apply_calls: int
    solver_vjp_calls: int
    closure_apply_calls: int
    closure_vjp_calls: int
    solver_vjp_min_cotangent_norm: float
    closure_vjp_min_cotangent_norm: float
    solver_vjp_input_paths: list[list[str]]
    closure_vjp_input_paths: list[list[str]]
    solver_transpose_sensitivity: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@contextlib.contextmanager
def tesseract_clients(*, use_images: bool) -> Iterator[tuple[object, object]]:
    """Yield solver/closure clients, guaranteeing image teardown on errors.

    Shared by the optimiser demo and the served training run: both need the
    same in-process-or-served client pair with the same teardown guarantee.
    """
    if use_images:
        clients = image_tesseract_clients()
        try:
            yield clients
        finally:
            teardown_image_clients(*clients)
        return
    yield local_tesseract_clients()


def _two_step_objective(
    solver: object,
    closure: object,
    candidate: jax.Array,
    initial_state: jax.Array,
    target_states: jax.Array,
) -> jax.Array:
    """Two-step rollout vorticity MSE, the locked hybrid rollout metric.

    The parameter gradient of this objective necessarily travels through both
    closure corrections and, for the first correction, through the second
    coarse solver step: omitting or zeroing the solver VJP removes a genuine
    term from the gradient.
    """
    rollout = composed_tesseract_rollout(
        solver,
        closure,
        candidate,
        initial_state,
        num_steps=DEMO_UNROLL,
        dt=DEMO_DT,
    )
    return vorticity_mse(rollout, target_states)


def run_optimiser_demo(
    *,
    use_images: bool = False,
    max_updates: int = DEMO_MAX_UPDATES,
) -> OptimiserDemoResult:
    """Apply up to ``max_updates`` Adam steps through the two Tesseracts.

    Runs on the locked calibrated regime (seed 0, vorticity amplitude 20,
    dt 0.002, two-step rollout) on genuine train-split DNS targets. The
    objective is the vorticity MSE between the two compiled rollout states
    and both filtered DNS targets, matching the project rollout metric. Every
    endpoint invocation is recorded at the client boundary; the demo requires
    the full composed-rollout evidence (solver and closure VJPs with the
    expected input paths and finite non-zero cotangents) so it cannot claim
    reverse mode through the solver without having measured it. If an update
    does not improve the rollout vorticity MSE, the parameters are reverted so
    an overshooting demo never ships a worse model. The report also records
    the solver-transpose sensitivity: the relative change in the parameter
    gradient when the solver VJP's ``omega`` cotangent is zeroed on the same
    clients, which measures how much of the gradient the solver transpose
    carries.
    """
    if isinstance(max_updates, bool) or not isinstance(max_updates, int):
        raise TypeError(
            f"max_updates must be an integer, got {type(max_updates).__name__}"
        )
    if max_updates <= 0:
        raise ValueError("max_updates must be positive")

    dns_config = DNSConfig(
        dt=DEMO_DT,
        vorticity_amplitude=DEMO_VORTICITY_AMPLITUDE,
    )
    reference = generate_reference_trajectory(
        DEMO_SEED,
        DEMO_UNROLL,
        split="train",
        config=dns_config,
    )
    target_states = reference.targets
    initial_state = reference.initial_coarse

    with tesseract_clients(use_images=use_images) as (solver, closure):
        solver = InstrumentedTesseract(solver)
        closure = InstrumentedTesseract(closure)
        params = jnp.asarray(initial_parameters(), dtype=jnp.float32)

        def objective(candidate: jax.Array) -> jax.Array:
            return _two_step_objective(
                solver,
                closure,
                candidate,
                initial_state,
                target_states,
            )

        loss_before = float(objective(params))

        optimiser = optax.adam(DEMO_LEARNING_RATE)
        optimiser_state = optimiser.init(params)
        candidate = params
        initial_gradient: jax.Array | None = None
        for _ in range(max_updates):
            gradient = jax.grad(objective)(candidate)
            if not bool(jnp.all(jnp.isfinite(gradient))):
                raise FloatingPointError("non-finite gradient in optimiser demo")
            if initial_gradient is None:
                initial_gradient = gradient
            updates, optimiser_state = optimiser.update(
                gradient,
                optimiser_state,
                candidate,
            )
            candidate = optax.apply_updates(candidate, updates)

        if initial_gradient is None:  # unreachable: max_updates >= 1 enforced
            raise AssertionError("no gradient computed in optimiser demo")
        zeroed_solver = InstrumentedTesseract(
            solver._wrapped,
            vjp_override=solver_omega_zeroing_vjp_override(solver._wrapped),
        )
        g_zero = jax.grad(
            lambda p: _two_step_objective(
                zeroed_solver,
                closure._wrapped,
                p,
                initial_state,
                target_states,
            )
        )(params)
        true_norm = float(jnp.linalg.norm(initial_gradient))
        if true_norm > 0.0:
            solver_transpose_sensitivity = float(
                jnp.linalg.norm(initial_gradient - g_zero) / true_norm
            )
        else:
            solver_transpose_sensitivity = 0.0
        violations = composition_invariant_violations(
            solver_apply_calls=solver.apply_calls,
            closure_apply_calls=closure.apply_calls,
            solver_vjp_calls=solver.vjp_calls,
            closure_vjp_calls=closure.vjp_calls,
            solver_vjp_input_paths=solver.vjp_input_paths,
            closure_vjp_input_paths=closure.vjp_input_paths,
            solver_vjp_min_cotangent_norm=(
                min(solver.vjp_cotangent_norms) if solver.vjp_cotangent_norms else 0.0
            ),
            closure_vjp_min_cotangent_norm=(
                min(closure.vjp_cotangent_norms) if closure.vjp_cotangent_norms else 0.0
            ),
        )
        if violations:
            raise RuntimeError(
                "the two-step parameter gradient did not exercise the full "
                "composition: " + "; ".join(violations)
            )
        loss_after = float(objective(candidate))
        loss_improved = loss_after < loss_before
        if loss_improved:
            parameter_update_norm = float(jnp.linalg.norm(candidate - params))
            accepted_updates = max_updates
        else:
            candidate = params
            loss_after = loss_before
            parameter_update_norm = 0.0
            accepted_updates = 0

        return OptimiserDemoResult(
            loss_before=loss_before,
            loss_after=loss_after,
            gradient_norm=float(jnp.linalg.norm(initial_gradient)),
            gradient_size=int(initial_gradient.size),
            gradient_finite=bool(jnp.all(jnp.isfinite(initial_gradient))),
            parameter_update_norm=parameter_update_norm,
            loss_improved=loss_improved,
            accepted_updates=accepted_updates,
            seed=DEMO_SEED,
            dt=DEMO_DT,
            learning_rate=DEMO_LEARNING_RATE,
            vorticity_amplitude=DEMO_VORTICITY_AMPLITUDE,
            use_images=use_images,
            unroll_steps=DEMO_UNROLL,
            objective=DEMO_OBJECTIVE,
            solver_apply_calls=solver.apply_calls,
            solver_vjp_calls=solver.vjp_calls,
            closure_apply_calls=closure.apply_calls,
            closure_vjp_calls=closure.vjp_calls,
            solver_vjp_min_cotangent_norm=(
                min(solver.vjp_cotangent_norms) if solver.vjp_cotangent_norms else 0.0
            ),
            closure_vjp_min_cotangent_norm=(
                min(closure.vjp_cotangent_norms) if closure.vjp_cotangent_norms else 0.0
            ),
            solver_vjp_input_paths=solver.vjp_input_paths,
            closure_vjp_input_paths=closure.vjp_input_paths,
            solver_transpose_sensitivity=solver_transpose_sensitivity,
        )


def write_optimiser_demo_report(result: OptimiserDemoResult, path: str | Path) -> Path:
    """Atomically write an optimiser demo result as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    temporary.replace(destination)
    return destination
