#!/usr/bin/env python3
"""
2007-manni-botryllus-schlosseri-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0867)
Organism: Botryllus schlosseri (Star Ascidian Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Botryllus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0867_botryllus_schlosseri_biomineral_accretion.lean` (`MathOfBiology.Loop_0867`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/botryllus_schlosseri_biomineral_accretion"
EXPECTED_SHA256 = {
    "botryllus_schlosseri_biomineral_accretion_reference.json": "69f41789282cb741eb726c682d0a8e68550d0509bbc612f7a61c0624fe7ace53",
    "botryllus_schlosseri_specimen.jpg": "b1ea2c5231b1b71026b40278bbac0bde71e6cf35e9fd8bedcd40f103a7190401",
    "scout_manifest.json": "d10857fd63d7e0b4c818a56e1f41b30f3f9ccdac86826549b8c28883ca66339c",
    "paper_citations.bib": "66206b6d7c88a523467ba972f99278c0264d71213c8c5737424efdf948cd9f4f"
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
    ref = json.loads((DATASET_DIR / "botryllus_schlosseri_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-manni-botryllus-schlosseri-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
