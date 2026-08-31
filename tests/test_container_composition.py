"""Containerised two-Tesseract composition smoke test (demo/submission path).

Requires a running Docker daemon and the two locally built images
(``coarse_solver:0.2.0`` and ``scalar_closure:0.2.0``, via ``make
build-tesseracts``). Skips cleanly when either is unavailable so the marker can
be selected on any machine without spurious failures.
"""

import shutil
import subprocess

import jax
import jax.numpy as jnp
import pytest

from tesseract_hybrid_closure.closure import initial_parameters, parameter_count
from tesseract_hybrid_closure.configs import DNSConfig
from tesseract_hybrid_closure.data import generate_reference_trajectory
from tesseract_hybrid_closure.losses import vorticity_mse
from tesseract_hybrid_closure.tesseract_components import (
    composed_tesseract_rollout,
    image_tesseract_clients,
    teardown_image_clients,
)
from tesseract_hybrid_closure.tesseract_demo import (
    DEMO_DT,
    DEMO_SEED,
    DEMO_UNROLL,
)
from tesseract_hybrid_closure.tesseract_instrumentation import (
    InstrumentedTesseract,
    composition_invariant_violations,
)

REQUIRED_IMAGES = ("coarse_solver:0.2.0", "scalar_closure:0.2.0")
DEMO_VORTICITY_AMPLITUDE = 20.0


def _docker_images() -> set[str] | None:
    """Return the set of local image tags, or None if Docker is unavailable."""
    if shutil.which("docker") is None:
        return None
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return set(result.stdout.split())


@pytest.mark.container
def test_containerised_two_step_composition_has_end_to_end_parameter_gradient():
    images = _docker_images()
    if images is None:
        pytest.skip("Docker daemon unavailable")
    missing = [name for name in REQUIRED_IMAGES if name not in images]
    if missing:
        pytest.skip(f"missing images {missing}; run `make build-tesseracts`")

    params = jnp.asarray(initial_parameters())
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

    solver, closure = image_tesseract_clients()
    try:
        solver = InstrumentedTesseract(solver)
        closure = InstrumentedTesseract(closure)

        def loss(candidate):
            rollout = composed_tesseract_rollout(
                solver,
                closure,
                candidate,
                reference.initial_coarse,
                num_steps=DEMO_UNROLL,
                dt=DEMO_DT,
            )
            return vorticity_mse(rollout, target_states)

        value, gradient = jax.value_and_grad(loss)(params)
    finally:
        teardown_image_clients(solver, closure)

    assert bool(jnp.isfinite(value))
    assert gradient.shape == params.shape
    assert gradient.size == parameter_count()
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
    # The served composition must exercise both VJPs with the expected input
    # paths and finite non-zero cotangents, exactly like the in-process path.
    assert (
        composition_invariant_violations(
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
        == []
    )
