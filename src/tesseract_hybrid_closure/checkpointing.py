"""Versioned, atomic checkpoints for trusted local training runs."""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import tempfile
from pathlib import Path

CHECKPOINT_FORMAT = "hybrid-closure-aposteriori-training"
CHECKPOINT_VERSION = 1

_HASH_CHUNK_BYTES = 1 << 20
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def validate_sha256_digest(digest: object) -> str:
    """Return the digest after strict format validation.

    A recorded checkpoint digest must be exactly 64 lowercase hex characters;
    anything else is rejected so a tampered report can never smuggle an
    unverifiable marker through path-based validation.
    """
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(
            "SHA-256 digest must be a 64-character lowercase hex string, "
            f"got {digest!r}"
        )
    return digest


def read_training_checkpoint_with_digest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict, str]:
    """Load a checkpoint together with the SHA-256 of its exact bytes.

    The file is read once in chunks: each chunk is fed to the streaming hash
    and collected, and the checkpoint is unpickled from the identical byte
    stream that was hashed. When ``expected_sha256`` is given the recorded
    digest is format-validated and the file must match it byte-for-byte
    before anything is unpickled.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"training checkpoint not found: {source}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
            chunks.append(chunk)
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        validate_sha256_digest(expected_sha256)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"checkpoint SHA-256 mismatch for {source}: recorded "
                f"digest {expected_sha256}, actual {actual_sha256}"
            )
    checkpoint = pickle.loads(b"".join(chunks))
    return _validate_checkpoint(checkpoint), actual_sha256


def _validate_checkpoint(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("training checkpoint must contain a mapping")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported training checkpoint format")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported training checkpoint version")
    return checkpoint


def save_training_checkpoint(path: str | Path, payload: dict) -> Path:
    """Atomically create a checkpoint without overwriting an existing run."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        **payload,
    }

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)
    return destination


def load_training_checkpoint(path: str | Path) -> dict:
    """Load and validate a trusted local checkpoint."""
    checkpoint, _ = read_training_checkpoint_with_digest(path)
    return checkpoint
