"""Validated configuration for the differentiable hybrid core."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .constants import (
    COARSE_DT,
    COARSE_NUM_POINTS,
    DEFAULT_DIFFUSIVITY,
    DNS_NUM_POINTS,
    DOMAIN_EXTENT,
    ETDRK_ORDER,
    LEARNING_RATE,
    TEST_SEED_RANGE,
    TRAIN_SEED_RANGE,
    UNROLL_CURRICULUM,
    VALIDATION_SEED_RANGE,
)


@dataclass(frozen=True)
class SolverConfig:
    """Exponax coarse-vorticity stepper configuration."""

    num_points: int = COARSE_NUM_POINTS
    domain_extent: float = DOMAIN_EXTENT
    dt: float = COARSE_DT
    diffusivity: float = DEFAULT_DIFFUSIVITY
    order: int = ETDRK_ORDER

    def __post_init__(self) -> None:
        if self.num_points != COARSE_NUM_POINTS:
            raise ValueError(
                f"the locked solver requires a {COARSE_NUM_POINTS}² coarse grid"
            )
        for name in ("domain_extent", "dt", "diffusivity"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.order != ETDRK_ORDER:
            raise ValueError(f"the locked solver requires ETDRK order {ETDRK_ORDER}")


@dataclass(frozen=True)
class DNSConfig:
    """Exponax DNS reference-stepper configuration."""

    num_points: int = DNS_NUM_POINTS
    domain_extent: float = DOMAIN_EXTENT
    dt: float = COARSE_DT
    diffusivity: float = DEFAULT_DIFFUSIVITY
    order: int = ETDRK_ORDER
    vorticity_amplitude: float = 1.0

    def __post_init__(self) -> None:
        if self.num_points != DNS_NUM_POINTS:
            raise ValueError(
                f"the locked protocol requires a {DNS_NUM_POINTS}² DNS grid"
            )
        for name in ("domain_extent", "dt", "diffusivity", "vorticity_amplitude"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.order != ETDRK_ORDER:
            raise ValueError(f"the locked protocol requires ETDRK order {ETDRK_ORDER}")


@dataclass(frozen=True)
class TrainingConfig:
    """Locked optimiser/curriculum with explicit update counts."""

    updates_per_stage: tuple[int, int, int]
    learning_rate: float = LEARNING_RATE
    curriculum: tuple[int, int, int] = UNROLL_CURRICULUM

    def __post_init__(self) -> None:
        if len(self.updates_per_stage) != len(self.curriculum):
            raise ValueError("updates_per_stage must contain one count per stage")
        if any(int(count) <= 0 for count in self.updates_per_stage):
            raise ValueError("all stage update counts must be positive")
        if self.curriculum != UNROLL_CURRICULUM:
            raise ValueError(
                f"the locked protocol requires curriculum {UNROLL_CURRICULUM}"
            )
        if self.learning_rate != LEARNING_RATE:
            raise ValueError(
                f"the locked protocol requires Adam learning rate {LEARNING_RATE}"
            )
        if sum(self.updates_per_stage) > len(TRAIN_SEED_RANGE):
            raise ValueError("training updates exceed the disjoint training seed range")


def seed_range_for_split(split: str) -> range:
    """Return the locked, disjoint seed range for one data split."""
    ranges = {
        "train": TRAIN_SEED_RANGE,
        "validation": VALIDATION_SEED_RANGE,
        "test": TEST_SEED_RANGE,
    }
    try:
        return ranges[split]
    except KeyError as error:
        raise ValueError(f"unknown data split: {split!r}") from error


@dataclass(frozen=True)
class ComponentConfig:
    """Local or packaged Tesseract component references."""

    coarse_solver: str = "coarse_solver"
    scalar_closure: str = "scalar_closure"

    def __post_init__(self) -> None:
        for name in ("coarse_solver", "scalar_closure"):
            value = getattr(self, name)
            if not value or not value.strip():
                raise ValueError(f"{name} must be a non-empty component reference")
