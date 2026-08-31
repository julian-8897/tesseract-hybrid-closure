"""Load local Tesseract APIs without changing global import paths."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TESSERACTS_ROOT = REPO_ROOT / "tesseracts"


def load_tesseract_api(component: str, *, module_name: str | None = None) -> ModuleType:
    """Load one local component's ``tesseract_api.py`` under a unique name."""
    if not component or Path(component).name != component:
        raise ValueError(f"Invalid Tesseract component name: {component!r}")

    module_path = TESSERACTS_ROOT / component / "tesseract_api.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Tesseract API not found: {module_path}")

    resolved_name = module_name or f"_local_tesseract_{component}"
    spec = importlib.util.spec_from_file_location(resolved_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Tesseract API: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CONFIG_KEY = re.compile(
    r"^(?P<indent> *)(?P<key>[A-Za-z_][A-Za-z0-9_]*): *(?P<value>.*)$"
)


def read_component_config(component: str) -> dict[str, Any]:
    """Read ``name``, ``version`` and ``metadata`` from a component config.

    A deliberately small reader for the flat scalar fields this repo's
    ``tesseract_config.yaml`` files use, so served provenance stays consistent
    with what the build actually produces without pulling in a YAML
    dependency. Nested lists and deeper structures are skipped rather than
    parsed.
    """
    if not component or Path(component).name != component:
        raise ValueError(f"Invalid Tesseract component name: {component!r}")
    config_path = TESSERACTS_ROOT / component / "tesseract_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Tesseract config not found: {config_path}")

    top: dict[str, Any] = {}
    section: dict[str, str] | None = None
    for line in config_path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "-")):
            continue
        match = _CONFIG_KEY.match(line)
        if match is None:
            continue
        indent = len(match.group("indent"))
        key = match.group("key")
        value = match.group("value").strip().strip('"').strip("'")
        if indent == 0:
            section = {} if not value else None
            top[key] = section if section is not None else value
        elif indent == 2 and section is not None and value:
            section[key] = value
    for required in ("name", "version"):
        if not isinstance(top.get(required), str):
            raise ValueError(f"{config_path} does not declare a scalar {required!r}")
    metadata = top.get("metadata")
    top["metadata"] = metadata if isinstance(metadata, dict) else {}
    return top
