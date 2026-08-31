#!/usr/bin/env bash
# Build every locally configured Tesseract image.

set -euo pipefail

# Use the tesseract-core CLI from the project environment, not whatever
# `tesseract` binary happens to be first on PATH (e.g. the unrelated OCR engine).
TESSERACT=(uv run tesseract)

if ! "${TESSERACT[@]}" --version >/dev/null 2>&1; then
    printf '%s\n' "Error: tesseract-core CLI unavailable. Run 'uv sync --locked' first." >&2
    exit 1
fi

for tess_dir in tesseracts/*/; do
    printf 'Building %s\n' "${tess_dir}"
    "${TESSERACT[@]}" build "${tess_dir}"
done
