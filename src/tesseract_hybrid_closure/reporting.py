"""Small reporting helpers for reproducible smoke-test artefacts."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import SmokeResult


def write_smoke_report(result: SmokeResult, path: str | Path) -> Path:
    """Atomically write a smoke result as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    temporary.replace(destination)
    return destination
