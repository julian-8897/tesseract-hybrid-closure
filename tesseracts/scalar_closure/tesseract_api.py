# Copyright 2026 Julian Chan
# SPDX-License-Identifier: Apache-2.0

"""PyTorch scalar SGS-tendency closure Tesseract API."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32

from tesseract_hybrid_closure.torch_closure import (
    parameter_count,
    torch_forward,
    torch_vjp,
)


class InputSchema(BaseModel):
    """Inputs for the flattened-parameter PyTorch closure."""

    omega: Differentiable[Array[(1, 64, 64), Float32]] = Field(
        description="Coarse vorticity with channel-first shape (1, 64, 64)"
    )
    params_flat: Differentiable[Array[(None,), Float32]] = Field(
        description="Flattened PyTorch CNN parameters"
    )


class OutputSchema(BaseModel):
    """Scalar SGS tendency produced by the CNN."""

    tendency: Differentiable[Array[(1, 64, 64), Float32]] = Field(
        description="Learned scalar vorticity tendency"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the closure with parameters transported as a flat array."""
    return {
        "tendency": torch_forward(
            np.asarray(inputs.params_flat, dtype=np.float32),
            np.asarray(inputs.omega, dtype=np.float32),
        )
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Compute the PyTorch VJP used by ``tesseract-jax`` reverse mode."""
    supported_inputs = {"params_flat", "omega"}
    unsupported = vjp_inputs - supported_inputs
    if unsupported or vjp_outputs != {"tendency"}:
        raise NotImplementedError(
            "The closure VJP supports tendency with respect to params_flat and omega"
        )
    params_gradient, omega_gradient = torch_vjp(
        np.asarray(inputs.params_flat, dtype=np.float32),
        np.asarray(inputs.omega, dtype=np.float32),
        np.asarray(cotangent_vector["tendency"], dtype=np.float32),
    )
    gradients = {}
    if "params_flat" in vjp_inputs:
        gradients["params_flat"] = params_gradient
    if "omega" in vjp_inputs:
        gradients["omega"] = omega_gradient
    return gradients


def abstract_eval(abstract_inputs):
    """Return the fixed output shape and dtype after validating parameter length."""
    params = abstract_inputs.params_flat
    shape = params.get("shape") if isinstance(params, dict) else params.shape
    if shape[0] not in (None, parameter_count()):
        raise ValueError(
            f"params_flat must contain {parameter_count()} values, got shape {shape}"
        )
    return {"tendency": {"shape": (1, 64, 64), "dtype": "float32"}}
