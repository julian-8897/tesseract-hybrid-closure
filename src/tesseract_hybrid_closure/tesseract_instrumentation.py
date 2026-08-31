"""Client-side endpoint recording for Tesseract composition evidence."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract


def _joint_norm(cotangent_vector: dict[str, Any]) -> float:
    """Return the Euclidean norm of all cotangent entries in one VJP call."""
    norms = [
        float(np.linalg.norm(np.asarray(value).ravel()))
        for value in cotangent_vector.values()
    ]
    return float(np.linalg.norm(norms))


def solver_omega_zeroing_vjp_override(wrapped: Tesseract):
    """Return a solver VJP override that zeroes only the ``omega`` cotangent.

    The override calls the real endpoint and then replaces the ``omega`` entry
    of the returned input cotangents with zeros, leaving every other entry
    untouched. A gradient computed under the override differs from the true
    gradient exactly by the solver-transpose contribution, which is the
    load-bearing cross-component path of a multi-step rollout objective.
    """

    def override(
        inputs,
        vjp_inputs,
        vjp_outputs,
        cotangent_vector,
        run_id=None,
    ):
        result = wrapped.vector_jacobian_product(
            inputs,
            vjp_inputs,
            vjp_outputs,
            cotangent_vector,
            run_id=run_id,
        )
        return {
            key: (jnp.zeros_like(value) if key == "omega" else value)
            for key, value in result.items()
        }

    return override


class InstrumentedTesseract(Tesseract):
    """Wrap a Tesseract client and record endpoint invocations.

    Counts ``apply``, ``vector_jacobian_product`` and
    ``jacobian_vector_product`` calls at the client boundary, which for a
    served image is one HTTP round trip per endpoint invocation. Records the
    requested VJP input paths and the cotangent norm of every VJP call, so a
    test can prove which endpoints a traced computation actually exercised.

    ``vjp_override`` replaces the VJP endpoint with a stub of the same
    signature (``inputs, vjp_inputs, vjp_outputs, cotangent_vector, run_id``)
    for gradient-path sensitivity tests.
    """

    def __init__(
        self,
        wrapped: Tesseract,
        *,
        vjp_override: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._vjp_override = vjp_override
        self.apply_calls = 0
        self.vjp_calls = 0
        self.jvp_calls = 0
        self.vjp_cotangent_norms: list[float] = []
        self.vjp_input_paths: list[list[str]] = []

    @property
    def openapi_schema(self) -> dict:
        """Proxy the wrapped client's OpenAPI schema."""
        return self._wrapped.openapi_schema

    @property
    def available_endpoints(self) -> list[str]:
        """Proxy the wrapped client's endpoint list."""
        return self._wrapped.available_endpoints

    def abstract_eval(self, abstract_inputs: dict) -> dict:
        """Proxy abstract evaluation for JAX tracing."""
        return self._wrapped.abstract_eval(abstract_inputs)

    def apply(self, inputs: dict, run_id: str | None = None) -> dict:
        """Count and forward one apply endpoint call."""
        self.apply_calls += 1
        return self._wrapped.apply(inputs, run_id=run_id)

    def vector_jacobian_product(
        self,
        inputs: dict,
        vjp_inputs: list[str],
        vjp_outputs: list[str],
        cotangent_vector: dict[str, Any],
        run_id: str | None = None,
    ) -> dict:
        """Count, record and forward one reverse-mode endpoint call."""
        self.vjp_calls += 1
        self.vjp_input_paths.append(sorted(vjp_inputs))
        self.vjp_cotangent_norms.append(_joint_norm(cotangent_vector))
        if self._vjp_override is not None:
            return self._vjp_override(
                inputs,
                vjp_inputs,
                vjp_outputs,
                cotangent_vector,
                run_id=run_id,
            )
        return self._wrapped.vector_jacobian_product(
            inputs,
            vjp_inputs,
            vjp_outputs,
            cotangent_vector,
            run_id=run_id,
        )

    def jacobian_vector_product(
        self,
        inputs: dict,
        jvp_inputs: list[str],
        jvp_outputs: list[str],
        tangent_vector: dict[str, Any],
        run_id: str | None = None,
    ) -> dict:
        """Count and forward one forward-mode endpoint call."""
        self.jvp_calls += 1
        return self._wrapped.jacobian_vector_product(
            inputs,
            jvp_inputs,
            jvp_outputs,
            tangent_vector,
            run_id=run_id,
        )

    def jacobian(
        self,
        inputs: dict,
        jac_inputs: list[str],
        jac_outputs: list[str],
        run_id: str | None = None,
    ) -> dict:
        """Forward one materialised-Jacobian endpoint call."""
        return self._wrapped.jacobian(
            inputs,
            jac_inputs,
            jac_outputs,
            run_id=run_id,
        )

    def teardown(self) -> None:
        """Tear down the wrapped client."""
        return self._wrapped.teardown()

    def __getattr__(self, name: str) -> Any:
        """Fall back to the wrapped client for any other attribute."""
        return getattr(self._wrapped, name)


def composition_invariant_violations(
    *,
    solver_apply_calls: int,
    closure_apply_calls: int,
    solver_vjp_calls: int,
    closure_vjp_calls: int,
    solver_vjp_input_paths: list[list[str]],
    closure_vjp_input_paths: list[list[str]],
    solver_vjp_min_cotangent_norm: float,
    closure_vjp_min_cotangent_norm: float,
) -> list[str]:
    """Return human-readable violations of the composed-rollout evidence invariants.

    A two-step rollout-loss parameter gradient through the composition must
    exercise the solver VJP at least once and the closure VJP at least twice
    (one per correction), every solver VJP must differentiate its ``omega``
    input, every closure VJP must differentiate ``params_flat`` and at least
    one must also differentiate ``omega`` (the path that carries the solver
    transpose back into the closure), and every VJP must have carried a
    finite, non-zero cotangent. Call counts are bounded below, never pinned
    exactly, so the evidence stays robust to trace pruning and evaluation
    count changes.
    """
    violations: list[str] = []
    if solver_apply_calls < 2:
        violations.append(f"solver apply calls {solver_apply_calls} < 2")
    if closure_apply_calls < 2:
        violations.append(f"closure apply calls {closure_apply_calls} < 2")
    if solver_vjp_calls < 1:
        violations.append(f"solver VJP calls {solver_vjp_calls} < 1")
    if closure_vjp_calls < 2:
        violations.append(f"closure VJP calls {closure_vjp_calls} < 2")
    if not solver_vjp_input_paths or not all(
        "omega" in path for path in solver_vjp_input_paths
    ):
        violations.append(
            f"solver VJP paths {solver_vjp_input_paths!r} do not all include omega"
        )
    if (
        not closure_vjp_input_paths
        or not all("params_flat" in path for path in closure_vjp_input_paths)
        or not any("omega" in path for path in closure_vjp_input_paths)
    ):
        violations.append(
            "closure VJP paths "
            f"{closure_vjp_input_paths!r} need params_flat everywhere and "
            "omega at least once"
        )
    for label, norm in (
        ("solver", solver_vjp_min_cotangent_norm),
        ("closure", closure_vjp_min_cotangent_norm),
    ):
        if not math.isfinite(norm) or norm <= 0.0:
            violations.append(
                f"{label} VJP min cotangent norm {norm!r} is not finite and > 0"
            )
    return violations
