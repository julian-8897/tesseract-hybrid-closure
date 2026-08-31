"""Integrity checks for the browser-only marimo walkthrough data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = REPO_ROOT / "notebooks" / "public"
METADATA_PATH = PUBLIC_DIR / "rollout_seed10000.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_browser_walkthrough_data_matches_manifest() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    assert metadata["schema"] == "marimo-browser-walkthrough-data"
    assert metadata["version"] == 2
    assert metadata["split"] == "validation"
    assert metadata["seed"] == 10000
    assert metadata["num_steps"] == 30
    assert metadata["checkpoint_sha256"] == (
        "8ed5b36fac902e61fcd3a1749f727f6928c93f3b7303e5781f5f66ce86c2e9b7"
    )

    arrays = {}
    for label, record in metadata["arrays"].items():
        path = PUBLIC_DIR / record["path"]
        values = np.load(path, allow_pickle=False)
        assert list(values.shape) == record["shape"] == [30, 1, 64, 64]
        assert str(values.dtype) == record["dtype"] == "float32"
        assert np.all(np.isfinite(values))
        assert _sha256(path) == record["sha256"]
        arrays[label] = values

    filtered_dns = arrays["filtered_dns"]
    learned_mse = float(np.mean((arrays["learned_closure"] - filtered_dns) ** 2))
    baseline_mse = float(np.mean((arrays["no_closure"] - filtered_dns) ** 2))

    assert learned_mse == pytest.approx(
        metadata["rollout_mse"]["learned_closure"], abs=1.0e-12
    )
    assert baseline_mse == pytest.approx(
        metadata["rollout_mse"]["no_closure"], abs=1.0e-12
    )
    assert learned_mse < baseline_mse

    # The stamped evidence must match the tracked reports exactly, so the
    # browser page can never drift from the sealed numbers.
    metrics = json.loads(
        (REPO_ROOT / "docs" / "results" / "final-metrics.json").read_text(
            encoding="utf-8"
        )
    )
    selected = metrics["selection"]["candidates"]["aposteriori-700"]
    evidence = metadata["evidence"]["selected_32_seed_validation_mse"]
    assert evidence["mean_vorticity_mse"] == selected["mean_vorticity_mse"]
    assert evidence["source"] == "docs/results/final-metrics.json"

    demo = json.loads(
        (REPO_ROOT / "docs" / "results" / "container-optimiser-demo.json").read_text(
            encoding="utf-8"
        )
    )
    served = metadata["evidence"]["served_demo"]
    assert served["gradient_size"] == demo["gradient_size"]
    assert served["gradient_finite"] is demo["gradient_finite"]
    assert (
        served["solver_transpose_sensitivity"] == demo["solver_transpose_sensitivity"]
    )
    assert served["solver_vjp_calls"] == demo["solver_vjp_calls"]
    assert served["closure_vjp_calls"] == demo["closure_vjp_calls"]
    assert served["source"] == "docs/results/container-optimiser-demo.json"
