"""Radially binned energy and enstrophy spectra for coarse vorticity fields.

A diagnostic only. Nothing here participates in training, model selection, or
the sealed test evaluation; it exists to show *where in wavenumber* the learned
closure changes the flow, which a vorticity-MSE alone cannot report.

Conventions, for a ``2π``-periodic square domain so that the physical
wavenumber equals the integer mode index:

- ``omega_hat = fft2(omega) / N**2``, so ``mean(omega**2) = sum_k |omega_hat|**2``;
- enstrophy density ``Z = 0.5 * mean(omega**2) = sum_k 0.5 * |omega_hat|**2``;
- in two dimensions ``omega = -laplacian(psi)`` and ``u = curl(psi)``, so
  ``|u_hat|**2 = |omega_hat|**2 / |k|**2`` and the energy density is
  ``E = 0.5 * mean(|u|**2) = sum_{k != 0} 0.5 * |omega_hat|**2 / |k|**2``.

Every mode is assigned to the shell ``round(|k|)``, so summing a returned
spectrum over all shells reproduces the corresponding total exactly. That
identity is the module's main invariant and is pinned by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import COARSE_NUM_POINTS, DOMAIN_EXTENT


@dataclass(frozen=True)
class RadialSpectra:
    """Shell-binned energy and enstrophy spectra.

    ``wavenumber`` has shape ``(S,)``; ``energy`` and ``enstrophy`` have shape
    ``(..., S)``, carrying through any leading batch or time axes of the input
    field.
    """

    wavenumber: np.ndarray
    energy: np.ndarray
    enstrophy: np.ndarray

    def __post_init__(self) -> None:
        if self.wavenumber.ndim != 1:
            raise ValueError("wavenumber must be one-dimensional")
        for name in ("energy", "enstrophy"):
            values = getattr(self, name)
            if values.shape[-1] != self.wavenumber.shape[0]:
                raise ValueError(
                    f"{name} last axis {values.shape[-1]} does not match "
                    f"{self.wavenumber.shape[0]} wavenumber shells"
                )

    def truncated(self, max_wavenumber: int) -> RadialSpectra:
        """Keep shells ``1 <= k <= max_wavenumber``, dropping the mean mode.

        Shells above ``N // 2`` hold only the partially resolved corners of the
        Fourier square, so plots use this rather than the full radial extent.
        """
        if max_wavenumber < 1:
            raise ValueError("max_wavenumber must be at least 1")
        keep = (self.wavenumber >= 1) & (self.wavenumber <= max_wavenumber)
        return RadialSpectra(
            wavenumber=self.wavenumber[keep],
            energy=self.energy[..., keep],
            enstrophy=self.enstrophy[..., keep],
        )


def _shell_index(num_points: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Radial shell index, squared wavenumber, and shell count for an N² grid."""
    modes = np.fft.fftfreq(num_points) * num_points
    kx = modes[:, None]
    ky = modes[None, :]
    squared = kx**2 + ky**2
    shell = np.rint(np.sqrt(squared)).astype(np.int64)
    return shell, squared, int(shell.max()) + 1


def radial_spectra(
    omega: np.ndarray,
    *,
    domain_extent: float = DOMAIN_EXTENT,
) -> RadialSpectra:
    """Energy and enstrophy spectra of a vorticity field or stack of fields.

    ``omega`` may carry any leading axes (time, seed, channel); the last two
    axes must be the square spatial grid. Returned spectra keep the leading
    axes and append a shell axis.
    """
    values = np.asarray(omega, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("omega must have at least two spatial axes")
    num_points = values.shape[-1]
    if values.shape[-2] != num_points:
        raise ValueError(
            f"omega must use a square spatial grid, got {values.shape[-2:]}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("omega must be finite")
    if not np.isfinite(domain_extent) or domain_extent <= 0.0:
        raise ValueError("domain_extent must be finite and positive")

    # Physical wavenumbers for a domain of extent L are 2*pi*n/L; the locked
    # 2*pi domain makes that the integer mode index, but do not assume it.
    scale = 2.0 * np.pi / float(domain_extent)
    shell, squared_modes, num_shells = _shell_index(num_points)
    squared_wavenumber = squared_modes * scale**2

    omega_hat = np.fft.fft2(values, axes=(-2, -1)) / num_points**2
    power = np.abs(omega_hat) ** 2

    enstrophy_density = 0.5 * power
    with np.errstate(divide="ignore", invalid="ignore"):
        energy_density = np.where(
            squared_wavenumber > 0.0,
            0.5 * power / squared_wavenumber,
            0.0,
        )

    leading = values.shape[:-2]
    flat_shell = shell.reshape(-1)

    def bin_shells(density: np.ndarray) -> np.ndarray:
        # (B, M) modes -> (S, B) accumulator -> (..., S), so every mode lands
        # in exactly one shell and the sum over shells is conserved.
        flat = density.reshape(-1, num_points**2)
        accumulator = np.zeros((num_shells, flat.shape[0]), dtype=np.float64)
        np.add.at(accumulator, flat_shell, flat.T)
        return accumulator.T.reshape(leading + (num_shells,))

    energy = bin_shells(energy_density)
    enstrophy = bin_shells(enstrophy_density)
    return RadialSpectra(
        wavenumber=np.arange(num_shells, dtype=np.int64),
        energy=energy,
        enstrophy=enstrophy,
    )


def total_enstrophy(omega: np.ndarray) -> np.ndarray:
    """``0.5 * mean(omega**2)`` over the spatial axes, for the Parseval check."""
    values = np.asarray(omega, dtype=np.float64)
    return 0.5 * np.mean(values**2, axis=(-2, -1))


def spectral_distance(
    model: np.ndarray,
    reference: np.ndarray,
    *,
    epsilon: float = 1.0e-30,
) -> float:
    """Mean absolute log-ratio between a model and reference spectrum.

    A scalar summary of spectral agreement that, unlike a difference of
    spectra, weights every shell equally instead of being dominated by the
    energy-containing scales. Lower is better; zero means identical shells.
    """
    model_values = np.asarray(model, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if model_values.shape != reference_values.shape:
        raise ValueError(
            f"shape mismatch: model {model_values.shape} vs "
            f"reference {reference_values.shape}"
        )
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    ratio = (model_values + epsilon) / (reference_values + epsilon)
    return float(np.mean(np.abs(np.log(ratio))))


def coarse_grid_max_wavenumber(num_points: int = COARSE_NUM_POINTS) -> int:
    """Highest fully resolved shell on an ``N²`` grid."""
    return num_points // 2


@dataclass(frozen=True)
class ErrorDecomposition:
    """Shell-binned split of an endpoint-field vorticity MSE into amplitude and phase parts.

    ``amplitude`` and ``phase`` have shape ``(..., S)`` and are both
    non-negative. Their total over all shells is exactly the mean squared
    vorticity difference between the two fields passed in, i.e. that
    endpoint field's vorticity MSE at the requested step. The reported
    rollout-prefix MSE averages over all states in the prefix, so only a
    one-step horizon makes the two coincide.
    """

    wavenumber: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray

    def __post_init__(self) -> None:
        for name in ("amplitude", "phase"):
            values = getattr(self, name)
            if values.shape[-1] != self.wavenumber.shape[0]:
                raise ValueError(
                    f"{name} last axis {values.shape[-1]} does not match "
                    f"{self.wavenumber.shape[0]} wavenumber shells"
                )

    @property
    def total(self) -> np.ndarray:
        """Summed squared error over all shells, equal to the vorticity MSE."""
        return self.amplitude.sum(axis=-1) + self.phase.sum(axis=-1)

    @property
    def phase_fraction(self) -> np.ndarray:
        """Share of the squared error attributable to phase disagreement."""
        total = self.total
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total > 0.0, self.phase.sum(axis=-1) / total, 0.0)

    def truncated(self, max_wavenumber: int) -> ErrorDecomposition:
        """Keep shells ``1 <= k <= max_wavenumber``, dropping the mean mode."""
        if max_wavenumber < 1:
            raise ValueError("max_wavenumber must be at least 1")
        keep = (self.wavenumber >= 1) & (self.wavenumber <= max_wavenumber)
        return ErrorDecomposition(
            wavenumber=self.wavenumber[keep],
            amplitude=self.amplitude[..., keep],
            phase=self.phase[..., keep],
        )


def error_decomposition(
    model: np.ndarray,
    reference: np.ndarray,
) -> ErrorDecomposition:
    """Split the mode-wise squared error into amplitude and phase parts.

    For one Fourier mode with model coefficient ``a`` and reference ``b``,

    ``|a - b|**2 = (|a| - |b|)**2 + 2|a||b|(1 - cos(arg a - arg b))``

    exactly. The first term is the error from getting the mode's magnitude
    wrong (dissipation or backscatter); the second is the error from getting
    its phase wrong (dispersion, and ultimately decorrelation). Both terms are
    non-negative, and summing them over every mode reproduces
    ``mean((model - reference)**2)`` by Parseval.

    This mode-wise magnitude/phase view follows the framing in Koehler,
    *From Numerical Simulators of PDEs to Neural Emulators and Back* (2026).
    The closed-form Fourier-multiplier analysis there applies to schemes that
    diagonalise in Fourier space; ours does not, being nonlinear in both the
    advection and the closure, so this is the empirical counterpart.
    """
    model_values = np.asarray(model, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if model_values.shape != reference_values.shape:
        raise ValueError(
            f"shape mismatch: model {model_values.shape} vs "
            f"reference {reference_values.shape}"
        )
    if model_values.ndim < 2:
        raise ValueError("fields must have at least two spatial axes")
    num_points = model_values.shape[-1]
    if model_values.shape[-2] != num_points:
        raise ValueError("fields must use a square spatial grid")
    if not (
        np.all(np.isfinite(model_values)) and np.all(np.isfinite(reference_values))
    ):
        raise ValueError("fields must be finite")

    model_hat = np.fft.fft2(model_values, axes=(-2, -1)) / num_points**2
    reference_hat = np.fft.fft2(reference_values, axes=(-2, -1)) / num_points**2

    model_magnitude = np.abs(model_hat)
    reference_magnitude = np.abs(reference_hat)
    amplitude_density = (model_magnitude - reference_magnitude) ** 2
    # 2|a||b|(1 - cos d) written as |a-b|^2 - (|a|-|b|)^2 would cancel
    # catastrophically for near-equal modes, so form it directly from the real
    # part of the cross-spectrum: 2(|a||b| - Re(a conj(b))).
    cross = np.real(model_hat * np.conjugate(reference_hat))
    phase_density = 2.0 * (model_magnitude * reference_magnitude - cross)
    # The identity guarantees non-negativity; floating point can still return a
    # tiny negative, which would be meaningless as an error contribution.
    phase_density = np.maximum(phase_density, 0.0)

    shell, _, num_shells = _shell_index(num_points)
    leading = model_values.shape[:-2]
    flat_shell = shell.reshape(-1)

    def bin_shells(density: np.ndarray) -> np.ndarray:
        flat = density.reshape(-1, num_points**2)
        accumulator = np.zeros((num_shells, flat.shape[0]), dtype=np.float64)
        np.add.at(accumulator, flat_shell, flat.T)
        return accumulator.T.reshape(leading + (num_shells,))

    return ErrorDecomposition(
        wavenumber=np.arange(num_shells, dtype=np.int64),
        amplitude=bin_shells(amplitude_density),
        phase=bin_shells(phase_density),
    )
