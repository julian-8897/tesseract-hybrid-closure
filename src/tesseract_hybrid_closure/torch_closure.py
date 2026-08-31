"""PyTorch scalar SGS-tendency CNN and flattened-parameter VJP."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from .constants import (
    CLOSURE_DEPTH,
    CLOSURE_HIDDEN_CHANNELS,
    CLOSURE_IN_CHANNELS,
    CLOSURE_KERNEL_SIZE,
    CLOSURE_OUT_CHANNELS,
    CLOSURE_SEED,
)


class ScalarTendencyCNN(nn.Module):
    """Ten-layer non-local CNN with periodic padding."""

    def __init__(self) -> None:
        super().__init__()
        padding = CLOSURE_KERNEL_SIZE // 2
        channels = [CLOSURE_IN_CHANNELS]
        channels.extend([CLOSURE_HIDDEN_CHANNELS] * (CLOSURE_DEPTH - 1))
        channels.append(CLOSURE_OUT_CHANNELS)
        self.layers = nn.ModuleList(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=CLOSURE_KERNEL_SIZE,
                padding=padding,
                padding_mode="circular",
            )
            for in_channels, out_channels in zip(
                channels[:-1], channels[1:], strict=True
            )
        )

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        """Map batched vorticity to a scalar SGS tendency."""
        hidden = omega
        for layer in self.layers[:-1]:
            hidden = torch.relu(layer(hidden))
        return self.layers[-1](hidden)


@lru_cache(maxsize=1)
def _reference_model() -> ScalarTendencyCNN:
    torch.manual_seed(CLOSURE_SEED)
    return ScalarTendencyCNN().eval()


def parameter_count() -> int:
    """Return the number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in _reference_model().parameters())


def initial_parameters() -> np.ndarray:
    """Return deterministic flattened float32 closure parameters."""
    arrays = [
        parameter.detach().cpu().numpy().reshape(-1)
        for parameter in _reference_model().parameters()
    ]
    return np.concatenate(arrays).astype(np.float32)


def _parameter_mapping(params_flat: torch.Tensor) -> dict[str, torch.Tensor]:
    model = _reference_model()
    expected = parameter_count()
    if params_flat.ndim != 1 or params_flat.numel() != expected:
        raise ValueError(
            f"params_flat must have shape ({expected},), got {tuple(params_flat.shape)}"
        )

    mapping: dict[str, torch.Tensor] = {}
    start = 0
    for name, parameter in model.named_parameters():
        stop = start + parameter.numel()
        mapping[name] = params_flat[start:stop].reshape(parameter.shape)
        start = stop
    return mapping


def _validate_omega(omega: np.ndarray) -> None:
    if omega.ndim != 3 or omega.shape[0] != CLOSURE_IN_CHANNELS:
        raise ValueError(f"omega must have shape (1, N, N), got {omega.shape}")
    if omega.shape[1] != omega.shape[2]:
        raise ValueError(f"omega spatial dimensions must be square, got {omega.shape}")
    if omega.dtype != np.float32:
        raise TypeError(f"omega must have dtype float32, got {omega.dtype}")
    if not np.isfinite(omega).all():
        raise ValueError("omega must contain only finite values")


def torch_forward(params_flat: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Evaluate the PyTorch closure from NumPy boundary values."""
    params_array = np.asarray(params_flat, dtype=np.float32)
    omega_array = np.asarray(omega)
    _validate_omega(omega_array)

    params_tensor = torch.from_numpy(np.array(params_array, copy=True))
    omega_tensor = torch.from_numpy(np.array(omega_array, copy=True)).unsqueeze(0)
    with torch.no_grad():
        tendency = functional_call(
            _reference_model(), _parameter_mapping(params_tensor), (omega_tensor,)
        )
    return tendency.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def torch_vjp(
    params_flat: np.ndarray, omega: np.ndarray, cotangent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the PyTorch VJP with respect to parameters and vorticity."""
    params_array = np.asarray(params_flat, dtype=np.float32)
    omega_array = np.asarray(omega)
    cotangent_array = np.asarray(cotangent, dtype=np.float32)
    _validate_omega(omega_array)
    if cotangent_array.shape != omega_array.shape:
        raise ValueError(
            "cotangent must match omega shape: "
            f"expected {omega_array.shape}, got {cotangent_array.shape}"
        )

    params_tensor = torch.from_numpy(np.array(params_array, copy=True)).requires_grad_()
    omega_tensor = (
        torch.from_numpy(np.array(omega_array, copy=True)).unsqueeze(0).requires_grad_()
    )
    tendency = functional_call(
        _reference_model(), _parameter_mapping(params_tensor), (omega_tensor,)
    )
    cotangent_tensor = torch.from_numpy(np.array(cotangent_array, copy=True)).unsqueeze(
        0
    )
    params_gradient, omega_gradient = torch.autograd.grad(
        tendency,
        (params_tensor, omega_tensor),
        grad_outputs=cotangent_tensor,
    )
    return (
        params_gradient.detach().cpu().numpy().astype(np.float32, copy=False),
        omega_gradient.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False),
    )
