"""Cross-check DM-DPP-01 sequence facts and Lean certificate (fix-012)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOP = ROOT / "morphoeval/lean/loops/Loop_prediction_DM_DPP_01_dpp_mobility_ratio.lean"
SEQUENCES = ROOT / "sequences/sequence_links.json"


def _dpp_entry() -> dict:
    data = json.loads(SEQUENCES.read_text(encoding="utf-8"))
    for row in data["resolved"]:
        if row.get("species") == "drosophila_melanogaster" and row.get("gene_symbol") == "dpp":
            return row
    raise AssertionError("drosophila_melanogaster dpp not found in sequence_links.json")


def test_sequence_links_match_dm_dpp_01_integers() -> None:
    row = _dpp_entry()
    ps = row["protein_state"]
    assert ps["precursor_length_aa"] == 588
    assert row["protein_length_aa"] == 588
    chain = ps["chains"][0]
    assert chain["start"] == 457
    assert chain["end"] == 588
    assert chain["length_aa"] == 132
    assert chain["end"] - chain["start"] + 1 == 132


def test_mobility_and_length_scale_integer_bounds() -> None:
    mature, precursor = 132, 588
    assert precursor * 607**3 < mature * 1000**3
    assert mature * 1000**3 < precursor * 608**3
    assert mature * 1000**3 < precursor * 610**3
    assert precursor * 779**6 < mature * 1000**6
    assert mature * 1000**6 <= precursor * 780**6


def test_lean_loop_has_no_sorry() -> None:
    assert LOOP.is_file(), f"missing Lean certificate: {LOOP}"
    text = LOOP.read_text(encoding="utf-8")
    stripped = re.sub(r"--.*", "", text)
    assert not re.search(r"(?<![A-Za-z_])sorry(?:Ax)?(?![A-Za-z_])", stripped)


def test_lean_kernel_check_when_available() -> None:
    lean = shutil.which("lean")
    if lean is None:
        elan = Path.home() / ".elan/bin/lean"
        if elan.is_file():
            lean = str(elan)
    if lean is None:
        return
    proc = subprocess.run(
        [lean, str(LOOP.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
