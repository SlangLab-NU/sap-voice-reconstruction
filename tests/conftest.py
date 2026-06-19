"""Shared pytest fixtures.

Real-data tests use a small fixture of approved SAP pairs under ``tests/fixtures/audio/``
(audio is gitignored — never committed, per the SAP data licence). If that audio is not
present (fresh clone / CI), the real-data tests **skip** cleanly; synthetic tests still run.
"""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST_DIR = Path("/projects/aanchan/data/manifests")


def _load_manifest():
    f = FIXTURES / "vtn_pairs.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def vtn_pairs():
    """Approved real SAP (source 16k, target 24k) pairs with resolved local paths.

    Skips the test if the fixture audio is absent.
    """
    resolved = []
    for p in _load_manifest():
        src, tgt = FIXTURES / p["source"], FIXTURES / p["target"]
        if src.exists() and tgt.exists():
            resolved.append({**p, "source": src, "target": tgt})
    if not resolved:
        pytest.skip("VTN fixture audio missing (tests/fixtures/audio/*.wav) — real-data tests skipped")
    return resolved


@pytest.fixture(scope="session")
def manifest_dir():
    """SAP Lhotse manifest dir; skips if the manifests or lhotse are absent."""
    if not (MANIFEST_DIR / "sap_recordings_val_source.jsonl.gz").exists():
        pytest.skip("SAP Lhotse manifests not present — manifest tests skipped")
    try:
        import lhotse  # noqa: F401
    except ImportError:
        pytest.skip("lhotse not installed — manifest tests skipped")
    return MANIFEST_DIR


@pytest.fixture(scope="session")
def val_cutsets(manifest_dir):
    """Loaded once: (source_cuts, target_cuts) for the val split."""
    from sap.data.manifest import load_pair_cutsets

    return load_pair_cutsets(manifest_dir, "val")
