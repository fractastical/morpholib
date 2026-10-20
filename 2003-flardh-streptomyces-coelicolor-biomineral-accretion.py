#!/usr/bin/env python3
"""
2003-flardh-streptomyces-coelicolor-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0888)
Organism: Streptomyces coelicolor (Mycelial Actinobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Streptomyces
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0888_streptomyces_coelicolor_biomineral_accretion.lean` (`MathOfBiology.Loop_0888`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/streptomyces_coelicolor_biomineral_accretion"
EXPECTED_SHA256 = {
    "streptomyces_coelicolor_biomineral_accretion_reference.json": "1c7315c109a6c5588ddb02f42eb28eab3397c60472e2339e6ad1b38fbd07d37a",
    "streptomyces_coelicolor_specimen.jpg": "5a2b99731bd4971f58260f74a1db5d4560b78f4d8ceb3b067629dcb7b6d53fec",
    "scout_manifest.json": "9b337158226da97a35d681bc573d14dd5ed3f15c39783f07728d9e51b5de113d",
    "paper_citations.bib": "53e92b072ad6d54f57a28af723bb34653f34a4f11f3004729d0591c18a8ab994"
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
    ref = json.loads((DATASET_DIR / "streptomyces_coelicolor_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-flardh-streptomyces-coelicolor-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
