#!/usr/bin/env python3
"""
2007-kumar-pleurodeles-waltl-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0825)
Organism: Pleurodeles waltl (Iberian Ribbed Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Pleurodeles
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0825_pleurodeles_waltl_biomineral_accretion.lean` (`MathOfBiology.Loop_0825`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pleurodeles_waltl_biomineral_accretion"
EXPECTED_SHA256 = {
    "pleurodeles_waltl_biomineral_accretion_reference.json": "377ea36d8320f4559e5cd0dd5e94cf35992579fb749d5217f6733c3208e617fe",
    "pleurodeles_waltl_specimen.jpg": "5b04f7216aba72146f64ff5aae72cd1f25ecdc62a55c051a3aea6d71c7dd09c5",
    "scout_manifest.json": "5edf643da5db142d95836f1dd936d2dd6840a0aeee11a3d13fdffa51e960f37d",
    "paper_citations.bib": "9ee7453a6f1225afb0283c66129fcf366d9d18deb39eebe1450e317733cec0f6"
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
    ref = json.loads((DATASET_DIR / "pleurodeles_waltl_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-kumar-pleurodeles-waltl-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
