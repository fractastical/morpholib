"""Pytest gate for Planform damaged-negative fixtures.

Each file under datasets/planform/damaged/ is a full planform_bioelectric_series_v1
fixture with exactly one injected fault. The manifest pins expected E_* error codes
and sha256 digests. These tests fail closed if a validator starts accepting a
negative or if manifest expectations drift from the files on disk.

Run:
    python3 -m pytest tests/test_planform_damaged_negatives.py -q

Related manual check:
    python3 models/planform_bioelectric.py --self-test
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DAMAGED_DIR = ROOT / "datasets" / "planform" / "damaged"
MANIFEST_PATH = DAMAGED_DIR / "manifest.json"

_spec = importlib.util.spec_from_file_location(
    "planform_bioelectric", ROOT / "models" / "planform_bioelectric.py"
)
planform = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(planform)


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.is_file(), f"missing {MANIFEST_PATH.relative_to(ROOT)}"
    return _load_manifest()


def test_manifest_lists_five_damaged_negatives(manifest: dict) -> None:
    files = manifest["files"]
    assert len(files) == 5
    codes = {item["expected_error_code"] for item in files}
    assert codes == {
        "E_FREQ_SUM",
        "E_NOTICE_MISSING",
        "E_UNKNOWN_MORPHOLOGY",
        "E_SHA_MISMATCH",
        "E_UNKNOWN_MECHANISM_CLASS",
    }


@pytest.mark.parametrize(
    "file_name,expected_code",
    [
        (item["file"], item["expected_error_code"])
        for item in _load_manifest()["files"]
    ],
    ids=[item["file"] for item in _load_manifest()["files"]],
)
def test_committed_damaged_negative_triggers_expected_code(
    file_name: str, expected_code: str
) -> None:
    item = next(i for i in _load_manifest()["files"] if i["file"] == file_name)
    assert planform.check_damaged_negative_item(DAMAGED_DIR, item) == []


def test_all_damaged_negatives_pass_manifest(manifest: dict) -> None:
    assert planform.check_all_damaged_negatives(DAMAGED_DIR) == []


def test_flipped_expected_error_code_fails_on_temp_copy(tmp_path: Path) -> None:
    """Negative control: wrong expected_error_code must fail the same check."""
    manifest = _load_manifest()
    item = dict(manifest["files"][0])
    src = DAMAGED_DIR / item["file"]
    dest = tmp_path / item["file"]
    shutil.copy2(src, dest)

    wrong_code = "E_SCHEMA_ID" if item["expected_error_code"] != "E_SCHEMA_ID" else "E_FREQ_SUM"
    item["expected_error_code"] = wrong_code

    failures = planform.check_damaged_negative_item(tmp_path, item)
    assert failures, "flipped expected_error_code should fail closed"
    assert any("expected" in msg for msg in failures)


def test_mutated_temp_copy_fails_even_with_correct_manifest_code(tmp_path: Path) -> None:
    """Negative control: corrupting a temp copy must not match pinned expectations."""
    manifest = _load_manifest()
    item = dict(manifest["files"][0])
    dest = tmp_path / item["file"]
    doc = json.loads((DAMAGED_DIR / item["file"]).read_text(encoding="utf-8"))
    doc["schema_id"] = "broken.schema"
    dest.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")

    failures = planform.check_damaged_negative_item(tmp_path, item)
    assert failures
