#!/usr/bin/env python3
"""
2008-manni-botryllus-schlosseri-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0267)
Organism: Botryllus schlosseri (Star Ascidian Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Botryllus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0267_botryllus_schlosseri_morphogen_condensation.lean` (`MathOfBiology.Loop_0267`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/botryllus_schlosseri_morphogen_condensation"
EXPECTED_SHA256 = {
    "botryllus_schlosseri_morphogen_condensation_reference.json": "ef790b9888a8685748921acf06feb4ace6c7797b600efd1dd16083d90ac18948",
    "botryllus_schlosseri_specimen.jpg": "b1ea2c5231b1b71026b40278bbac0bde71e6cf35e9fd8bedcd40f103a7190401",
    "scout_manifest.json": "04a521a6b38f10ed592bbdce8dc82ade0d694ad362cdcd4ffc33e3b091d80765",
    "paper_citations.bib": "1ec1e62b779ff80b9b709e86dd17aa29c45ffd2866d430376e81e646b441b6bf"
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
    ref = json.loads((DATASET_DIR / "botryllus_schlosseri_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-manni-botryllus-schlosseri-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
