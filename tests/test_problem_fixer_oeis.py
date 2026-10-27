"""OEIS A051023 Rule 30 center-column traceability (R30-OEIS-128-01).

Hermetic checks that the first 128 center-column bits match the vendored
fixture and the separate registration/resolution SHA-256 lock.

Run: python3 -m pytest tests/test_problem_fixer_oeis.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "datasets" / "conus" / "conus_textile_reference.json"
REG_PATH = ROOT / "predictions" / "registrations" / "R30-OEIS-128-01.json"
RES_PATH = ROOT / "predictions" / "resolutions" / "rule30-oeis.json"
MODEL_PATH = ROOT / "models" / "1984-wolfram-conus-textile-rule30.py"

EXPECTED_SHA256 = "a4d0748bb05e4a19738a0252b8f87b374385df59d0630c27dffb175d3fa98a5e"


def _load_model():
    spec = importlib.util.spec_from_file_location("conus_rule30", MODEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256_bits(bits: list[int]) -> str:
    s = "".join(str(int(b)) for b in bits)
    assert len(s) == 128
    return hashlib.sha256(s.encode("ascii")).hexdigest()


@pytest.fixture(scope="module")
def rule30_mod():
    return _load_model()


@pytest.fixture(scope="module")
def fixture_bits():
    data = json.loads(FIXTURE_PATH.read_text())
    return data["rule30_prize_benchmarks"]["oeis_a051023_first_128"]


def test_fixture_sha256_lock(fixture_bits):
    assert len(fixture_bits) == 128
    assert _sha256_bits(fixture_bits) == EXPECTED_SHA256


def test_center_column_matches_fixture(rule30_mod, fixture_bits):
    prize = rule30_mod.rule30_prize_diagnostics(steps=1024)
    computed = prize["center_col"][:128].astype(np.uint8)
    expected = np.array(fixture_bits, dtype=np.uint8)
    assert np.array_equal(computed, expected)


def test_registration_and_resolution_artifacts(fixture_bits):
    reg = json.loads(REG_PATH.read_text())
    res = json.loads(RES_PATH.read_text())
    assert reg["registration_id"] == "R30-OEIS-128-01"
    assert reg["expected_sequence"]["sha256_of_concatenated_bits"] == EXPECTED_SHA256
    row = res["resolutions"][0]
    assert row["prediction_id"] == "R30-OEIS-128-01"
    assert row["outcome_verdict"] == "confirmed"
    mv = row["machine_verdict"]
    assert mv["mismatch_count"] == 0
    assert mv["expected_sha256"] == mv["computed_sha256"] == EXPECTED_SHA256
