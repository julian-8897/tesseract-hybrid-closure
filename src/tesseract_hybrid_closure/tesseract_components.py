"""Composition through the public Tesseract and ``tesseract-jax`` APIs."""

from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

from .constants import COARSE_DT

REPO_ROOT = Path(__file__).resolve().parents[2]


def local_tesseract_clients() -> tuple[Tesseract, Tesseract]:
    """Return in-process clients backed by the two packaged API modules."""
    solver = Tesseract.from_tesseract_api(
        REPO_ROOT / "tesseracts/coarse_solver/tesseract_api.py"
    )
    closure = Tesseract.from_tesseract_api(
        REPO_ROOT / "tesseracts/scalar_closure/tesseract_api.py"
    )
    return solver, closure


def image_tesseract_clients(
    solver_image: str = "coarse_solver:0.2.0",
    closure_image: str = "scalar_closure:0.2.0",
) -> tuple[Tesseract, Tesseract]:
    """Start containerised clients for the demo composition path."""
    solver = Tesseract.from_image(solver_image)
    solver.serve()
    try:
        closure = Tesseract.from_image(closure_image)
        closure.serve()
    except Exception:
        solver.teardown()
        raise
    return solver, closure


def teardown_image_clients(*clients: Tesseract) -> None:
    """Stop clients created with :func:`image_tesseract_clients`."""
    for client in reversed(clients):
        client.teardown()


def _validated_dt(dt: float) -> float:
    """Coerce and validate a non-differentiable solver timestep."""
    resolved = float(dt)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}")
    return resolved


def composed_tesseract_step(
    solver: Tesseract,
    closure: Tesseract,
    params_flat: jax.Array,
    omega: jax.Array,
    dt: float = COARSE_DT,
) -> jax.Array:
    """Compose solver and closure Tesseracts with end-to-end reverse mode."""
    resolved_dt = _validated_dt(dt)
    base_state = apply_tesseract(solver, {"omega": omega, "dt": resolved_dt})[
        "omega_next"
    ]
    tendency = apply_tesseract(
        closure,
        {"omega": base_state, "params_flat": params_flat},
    )["tendency"]
    return base_state + jnp.asarray(resolved_dt, dtype=omega.dtype) * tendency


def composed_tesseract_rollout(
    solver: Tesseract,
    closure: Tesseract,
    params_flat: jax.Array,
    initial_state: jax.Array,
    num_steps: int,
    dt: float = COARSE_DT,
) -> jax.Array:
    """Roll out successive composed hybrid steps, returning every state.

    The gradient of a loss on the later states flows back through the solver
    step of each earlier correction, so a parameter gradient here necessarily
    exercises the solver VJP endpoint, not just the closure VJP.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int):
        raise TypeError(f"num_steps must be an integer, got {type(num_steps).__name__}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    state = initial_state
    states = []
    for _ in range(num_steps):
        state = composed_tesseract_step(
            solver,
            closure,
            params_flat,
            state,
            dt=dt,
        )
        states.append(state)
    return jnp.stack(states)
