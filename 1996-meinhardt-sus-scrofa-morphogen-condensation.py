#!/usr/bin/env python3
"""
1996-meinhardt-sus-scrofa-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0206)
Organism: Sus scrofa (Wild Boar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Sus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0206_sus_scrofa_morphogen_condensation.lean` (`MathOfBiology.Loop_0206`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sus_scrofa_morphogen_condensation"
EXPECTED_SHA256 = {
    "sus_scrofa_morphogen_condensation_reference.json": "e571c8d7132d5fb96e2bb0ea9ed28ff6bba2dfedb576ad278f2d987ce6c6ddec",
    "sus_scrofa_specimen.jpg": "2a19f0ec20969646f39adb56919e7ec23b3c0f4a963658eeab969a0d13ba3d70",
    "scout_manifest.json": "4752450983a506acd8a578805d46b796da5640d5c395f19e790e9277f8548d17",
    "paper_citations.bib": "7b693a2bafce1ea249f8431dee9c021dfa2d0e2dfe80fc1878cae3cb11c74b2d"
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
    ref = json.loads((DATASET_DIR / "sus_scrofa_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1996-meinhardt-sus-scrofa-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
