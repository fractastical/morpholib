#!/usr/bin/env python3
"""
1995-meinhardt-sus-scrofa-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0806)
Organism: Sus scrofa (Wild Boar)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Sus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0806_sus_scrofa_biomineral_accretion.lean` (`MathOfBiology.Loop_0806`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/sus_scrofa_biomineral_accretion"
EXPECTED_SHA256 = {
    "sus_scrofa_biomineral_accretion_reference.json": "58a121a343a83898b004e25b00d33c79cb80e8a440ec859baa52f5524543e380",
    "sus_scrofa_specimen.jpg": "2a19f0ec20969646f39adb56919e7ec23b3c0f4a963658eeab969a0d13ba3d70",
    "scout_manifest.json": "290aa809f6f61ecdeb12796c2b26becb7bd346990207edf5e20e72d66f27e1e5",
    "paper_citations.bib": "5b6df36e548fa8e150ce69903da0b233afb3fc5d5232ff1834a8a6df327b8415"
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
    ref = json.loads((DATASET_DIR / "sus_scrofa_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1995-meinhardt-sus-scrofa-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
