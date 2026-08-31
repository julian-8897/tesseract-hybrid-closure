"""Generate the browser walkthrough's deterministic validation trajectory.

This is presentation data only. It uses the selected checkpoint, one authorised
validation seed, and the locked 30-step regime. The three fields are staged and
published atomically together with the metadata, and existing outputs are never
replaced: a rerun after a partial or complete output set present will refuse to
touch it.

The metadata also stamps the tracked aggregate evidence the browser walkthrough
displays (the 32-seed validation MSE and the served-demo statistics), so the
browser page reads numbers from the tracked reports, not from literals in the
notebook.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import jax
import numpy as np

from tesseract_hybrid_closure.configs import DNSConfig, SolverConfig
from tesseract_hybrid_closure.data import generate_reference_trajectory
from tesseract_hybrid_closure.final_eval import load_candidate_params_with_digest
from tesseract_hybrid_closure.losses import closure_rollout, no_closure_rollout
from tesseract_hybrid_closure.solver import CoarseVorticityStepper

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (
    REPO_ROOT
    / "runs"
    / "w2-calibrated-a20-dt002-100x3"
    / "stage-unroll-30-updates-700.pkl"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "8ed5b36fac902e61fcd3a1749f727f6928c93f3b7303e5781f5f66ce86c2e9b7"
)
OUTPUT_DIR = REPO_ROOT / "notebooks" / "public"
FINAL_METRICS = REPO_ROOT / "docs" / "results" / "final-metrics.json"
CONTAINER_DEMO = REPO_ROOT / "docs" / "results" / "container-optimiser-demo.json"
SEED = 10000
NUM_STEPS = 30
EXPECTED_SHAPE = (NUM_STEPS, 1, 64, 64)
EXPECTED_DTYPE = np.float32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(path: Path, *, text: bool) -> Path:
    """Write to a same-directory temporary file and return it for publishing."""
    if path.exists():
        raise FileExistsError(f"refusing to replace existing demo data: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=text
    )
    return Path(temporary_name)


def _stage_npy(path: Path, values: np.ndarray) -> Path:
    temporary = _stage(path, text=False)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_json(path: Path, payload: dict) -> Path:
    temporary = _stage(path, text=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _evidence() -> dict:
    """Stamp the tracked aggregate evidence the walkthrough displays."""
    metrics = json.loads(FINAL_METRICS.read_text(encoding="utf-8"))
    demo = json.loads(CONTAINER_DEMO.read_text(encoding="utf-8"))
    selected = metrics["selection"]["candidates"]["aposteriori-700"]
    return {
        "selected_32_seed_validation_mse": {
            "mean_vorticity_mse": float(selected["mean_vorticity_mse"]),
            "source": "docs/results/final-metrics.json",
        },
        "served_demo": {
            "gradient_size": int(demo["gradient_size"]),
            "gradient_finite": bool(demo["gradient_finite"]),
            "solver_transpose_sensitivity": float(demo["solver_transpose_sensitivity"]),
            "solver_vjp_calls": int(demo["solver_vjp_calls"]),
            "closure_vjp_calls": int(demo["closure_vjp_calls"]),
            "source": "docs/results/container-optimiser-demo.json",
        },
    }


def main() -> None:
    """Generate and record one locked validation rollout."""
    destinations = {
        "filtered_dns": OUTPUT_DIR / "filtered_dns_seed10000.npy",
        "learned_closure": OUTPUT_DIR / "learned_closure_seed10000.npy",
        "no_closure": OUTPUT_DIR / "no_closure_seed10000.npy",
        "metadata": OUTPUT_DIR / "rollout_seed10000.json",
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to replace existing demo data: {joined}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint, params, checkpoint_digest = load_candidate_params_with_digest(
        CHECKPOINT,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    solver_config = SolverConfig(**checkpoint["solver_config"])
    dns_config = DNSConfig(**checkpoint["dns_config"])
    reference = generate_reference_trajectory(
        SEED,
        NUM_STEPS,
        split="validation",
        config=dns_config,
    )
    stepper = CoarseVorticityStepper(solver_config)
    learned = np.asarray(
        jax.device_get(
            closure_rollout(stepper, params, reference.initial_coarse, NUM_STEPS)
        ),
        dtype=EXPECTED_DTYPE,
    )
    no_closure = np.asarray(
        jax.device_get(
            no_closure_rollout(stepper, reference.initial_coarse, NUM_STEPS)
        ),
        dtype=EXPECTED_DTYPE,
    )
    filtered_dns = np.asarray(jax.device_get(reference.targets), dtype=EXPECTED_DTYPE)

    arrays = {
        "filtered_dns": filtered_dns,
        "learned_closure": learned,
        "no_closure": no_closure,
    }
    for label, values in arrays.items():
        if values.shape != EXPECTED_SHAPE:
            raise ValueError(
                f"{label} has shape {values.shape}, expected {EXPECTED_SHAPE}"
            )
        if values.dtype != EXPECTED_DTYPE:
            raise ValueError(f"{label} has dtype {values.dtype}, expected float32")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} contains non-finite values")

    evidence = _evidence()
    rollout_mse = {
        "learned_closure": float(np.mean((learned - filtered_dns) ** 2)),
        "no_closure": float(np.mean((no_closure - filtered_dns) ** 2)),
    }
    payload = {
        "schema": "marimo-browser-walkthrough-data",
        "version": 2,
        "split": "validation",
        "seed": SEED,
        "num_steps": NUM_STEPS,
        "checkpoint": str(CHECKPOINT.relative_to(REPO_ROOT)),
        "checkpoint_sha256": checkpoint_digest,
        "completed_updates": int(checkpoint["completed_updates"]),
        "solver_config": checkpoint["solver_config"],
        "dns_config": checkpoint["dns_config"],
        "arrays": {
            label: {
                "path": str(destinations[label].relative_to(OUTPUT_DIR)),
                "shape": list(EXPECTED_SHAPE),
                "dtype": "float32",
            }
            for label in ("filtered_dns", "learned_closure", "no_closure")
        },
        "rollout_mse": rollout_mse,
        "evidence": evidence,
        "note": (
            "Presentation data for the browser-only walkthrough. Generated from "
            "one validation seed; not a new reported evaluation result."
        ),
    }

    # Stage every output, then publish the whole set. A failure before the
    # publish loop leaves no trace; a failure during it removes whatever was
    # already published, so a rerun can start clean.
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for label, values in arrays.items():
            staged[destinations[label]] = _stage_npy(destinations[label], values)
        for label in ("filtered_dns", "learned_closure", "no_closure"):
            payload["arrays"][label]["sha256"] = _sha256(staged[destinations[label]])
        staged[destinations["metadata"]] = _stage_json(
            destinations["metadata"], payload
        )
        for destination, temporary in staged.items():
            temporary.replace(destination)
            published.append(destination)
    except Exception:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for destination in published:
            destination.unlink(missing_ok=True)
        raise

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
