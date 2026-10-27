"""Regression tests for fix-006: Turing audit script path resolution and measurement completeness."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def measure_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "measure_outputs_fix006", ROOT / "morphoeval" / "measure_outputs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference_sources(rows: list) -> str:
    return " ".join(str(x.get("reference_source", "")) for x in rows)


def test_turing_measurements_complete_with_models_script(measure_mod):
    gif = ROOT / "docs/media/turing_gray_scott.gif"
    if not gif.exists():
        pytest.skip("turing_gray_scott.gif not present")
    script = measure_mod.resolve_script_path("models/1952-turing-morpho.py")
    rows = measure_mod.measure_turing(gif, script=script)
    assert len(rows) >= 10
    src = _reference_sources(rows)
    assert "Pearson 1993" in src
    assert "Kondo" in src


def test_legacy_basename_resolves_under_models(measure_mod):
    legacy = measure_mod.resolve_script_path("1952-turing-morpho.py")
    assert legacy == ROOT / "models" / "1952-turing-morpho.py"
    assert legacy.exists()


def test_missing_script_fails_visibly_not_silent_partial(measure_mod):
    gif = ROOT / "docs/media/turing_gray_scott.gif"
    if not gif.exists():
        pytest.skip("turing_gray_scott.gif not present")
    bogus = measure_mod.resolve_script_path("no-such-turing-script-1952.py")
    assert not bogus.exists()
    rows = measure_mod.measure_turing(gif, script=bogus)
    assert len(rows) == 5
    assert rows[-1]["name"] == "script-backed Turing/Pearson/specimen measurements"
    assert rows[-1]["value"] is None
    assert "skipped: script not found" in rows[-1]["note"]


def test_measure_outputs_sidecar_turing_audit(measure_mod):
    audit = ROOT / "morphoeval/audits/2026-10-19-turing-audit-v2.yaml"
    sidecar = audit.with_suffix(".measurements.yaml")
    if not audit.exists():
        pytest.skip("audit yaml missing")
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(ROOT / "morphoeval/measure_outputs.py"), str(audit)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr or r.stdout
    import yaml

    doc = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    rows = doc["measurements"]["turing-gray-scott"]
    assert len(rows) >= 10
    src = _reference_sources(rows)
    assert "Pearson 1993" in src and "Kondo" in src
