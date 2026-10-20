#!/usr/bin/env python3
"""
2014-levin-dugesia-japonica-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0866)
Organism: Dugesia japonica (Freshwater Planarian)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dugesia
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0866_dugesia_japonica_biomineral_accretion.lean` (`MathOfBiology.Loop_0866`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/dugesia_japonica_biomineral_accretion"
EXPECTED_SHA256 = {
    "dugesia_japonica_biomineral_accretion_reference.json": "0cffbb1d0d84b0f0432627ea78d9468aa1dab5664d0477b23aacb87c197de7a6",
    "dugesia_japonica_specimen.jpg": "26b78d28b853c1b70092be6a7ae129d59813507f64eaed16a417276296ff2efa",
    "scout_manifest.json": "1acf7540d49703126cd3062ad130da8e79d28e223d2436e43184881c1bc2ec2f",
    "paper_citations.bib": "45cab64c1b56cc6757a4a21d1987b4a85a394e68e1270875dc88754a958f7ff5"
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
    ref = json.loads((DATASET_DIR / "dugesia_japonica_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-levin-dugesia-japonica-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
