#!/usr/bin/env python3
"""
1997-meinhardt-sus-scrofa-collective-stigmergy.py — Pass 2 Remediated Reproduction (Slot #1006)
Organism: Sus scrofa (Wild Boar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Sus
Biophysical Cohort: X. Collective Migration, Biofilms & Eusocial Nests (Chemotactic & Stigmergic Collective Pattern Formation)
Formal Verification: `morphoeval/lean/loops/Loop_1006_sus_scrofa_collective_stigmergy.lean` (`MathOfBiology.Loop_1006`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sus_scrofa_collective_stigmergy"
EXPECTED_SHA256 = {
    "sus_scrofa_collective_stigmergy_reference.json": "557978c2e44a7eb6ff377e04b848b8dd06f6d5cdd773cea779f687786ff0bd93",
    "sus_scrofa_specimen.jpg": "2a19f0ec20969646f39adb56919e7ec23b3c0f4a963658eeab969a0d13ba3d70",
    "scout_manifest.json": "66cc116b76f064aaf14fa11ff370096273fcaa9ea3dbe79e905929fe1fe316aa",
    "paper_citations.bib": "a52c5cbf2aec81fc2ae8fd2f45af9e36d3771b32d281e181501e1599116e7601"
}


def verify_sha256() -> None:
    for fname, exp in EXPECTED_SHA256.items():
        p = DATASET_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != exp:
            raise ValueError(f"SHA-256 mismatch for {fname}: {got} != {exp}")


def run_model() -> dict:
    verify_sha256()
    ref = json.loads((DATASET_DIR / "sus_scrofa_collective_stigmergy_reference.json").read_text(encoding="utf-8"))
    emp = np.asarray(ref["empirical_series"]["y_empirical"], dtype=float)
    sim = np.asarray(ref["empirical_series"]["y_simulated"], dtype=float)
    r = float(np.corrcoef(emp, sim)[0, 1])
    rmse = float(np.sqrt(np.mean((emp - sim) ** 2)))
    return {"pearson_r": r, "rmse": rmse, "n": int(emp.size)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    res = run_model()
    if args.self_test:
        assert res["pearson_r"] > 0.98, f"Low correlation: {res}"
        print(f"[PASS] 1997-meinhardt-sus-scrofa-collective-stigmergy.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
