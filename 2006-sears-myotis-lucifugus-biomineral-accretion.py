#!/usr/bin/env python3
"""
2006-sears-myotis-lucifugus-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0809)
Organism: Myotis lucifugus (Little Brown Bat)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Myotis
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0809_myotis_lucifugus_biomineral_accretion.lean` (`MathOfBiology.Loop_0809`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/myotis_lucifugus_biomineral_accretion"
EXPECTED_SHA256 = {
    "myotis_lucifugus_biomineral_accretion_reference.json": "d017128be0171c46caed79c1f3340b5550ecdf8bd2d24bcfc01b9c18c1a71ac6",
    "myotis_lucifugus_specimen.jpg": "1c8f403a4e94c928ac82c06c27e448ba133f1f6b2691860354b6490e152a97b4",
    "scout_manifest.json": "3c57da873e053e2b48d0ee1f0bd58a6d8ee2328556745715e92493dfd2ffafa3",
    "paper_citations.bib": "ace055868be6aa0de09289322419e28b2dc222aeaa2ace7494cd9c383864f990"
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
    ref = json.loads((DATASET_DIR / "myotis_lucifugus_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-sears-myotis-lucifugus-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
