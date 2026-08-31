# Attribution

This project is an independent Apache-2.0 repository. New work in this
repository is copyright 2026 `julian-8897`; the retained 2025 Pasteur Labs
notice covers the adapted upstream scaffold. Both attributions are recorded in
[`NOTICE`](NOTICE); [`LICENSE`](LICENSE) carries the full Apache-2.0 text.

The repository structure, local Tesseract loader, flattened-parameter transport,
and VJP endpoint pattern were adapted from:

- `julian-8897/tesseract-pinn-inverse-burgers`
- Local source commit: `b599280b3a81b50774c4eccfacd93ac79b8f842c`
- Original relevant files:
  - `src/burgers_inverse/component_loader.py`
  - `src/burgers_inverse/components.py`
  - `src/burgers_inverse/engine.py`
  - `src/burgers_inverse/checkpointing.py`
  - `src/burgers_inverse/reporting.py`
  - `src/burgers_inverse/configs.py`
  - `tesseracts/pinn_pytorch/tesseract_api.py`
  - `tesseracts/burgers_solver/tesseract_api.py`
  - `Makefile`, `buildall.sh`, and `pyproject.toml`

The reused source was licensed under Apache-2.0. The 2025 repository was not
modified when this repository was created.
