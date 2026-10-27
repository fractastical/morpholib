"""Focused tests for morphoeval/problem_scouts/verify_rule30_delphi_sync.py (fix-003)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "morphoeval" / "problem_scouts" / "verify_rule30_delphi_sync.py"


def test_delphi_sync_passes_on_repo_artifacts() -> None:
    proc = subprocess.run(
        [sys.executable, str(VERIFIER)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_delphi_sync_detects_open_problems_drift(tmp_path: Path) -> None:
    op_path = tmp_path / "open-problems.json"
    feed = json.loads((ROOT / "docs" / "open-problems.json").read_text(encoding="utf-8"))
    for p in feed["problems"]:
        if p["id"] == "OP-01":
            p["viability"]["pooled"] = 99.0
    op_path.write_text(json.dumps(feed), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--open-problems",
            str(op_path),
            "--quiet",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "OP-01" in proc.stderr


def test_yaml_tolerance_helper() -> None:
    from morphoeval.problem_scouts.verify_rule30_delphi_sync import yaml_within_tolerance

    assert yaml_within_tolerance(11.5184, 11.52)
    assert not yaml_within_tolerance(11.5184, 12.0)
