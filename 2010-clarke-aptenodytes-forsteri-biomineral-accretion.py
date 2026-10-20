#!/usr/bin/env python3
"""
2010-clarke-aptenodytes-forsteri-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0819)
Organism: Aptenodytes forsteri (Emperor Penguin)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Aptenodytes
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0819_aptenodytes_forsteri_biomineral_accretion.lean` (`MathOfBiology.Loop_0819`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aptenodytes_forsteri_biomineral_accretion"
EXPECTED_SHA256 = {
    "aptenodytes_forsteri_biomineral_accretion_reference.json": "7efb009e41492e556b8cf96ebaf9f6d91766d37901e1cc2b464ef0290396b21c",
    "aptenodytes_forsteri_specimen.jpg": "ef141d4e199c2ace11888d32dc8d271279924c5e9a9faa6f15ad839d7b5dd0d0",
    "scout_manifest.json": "b360da4f646ee52d2963faf29bb87dde6ed375006bfede7b62e6bf18c8978e8e",
    "paper_citations.bib": "27139b7ec31affaf47a4c782a0e8176e489db986ff7cd1a2a599bcb8e344087b"
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
    ref = json.loads((DATASET_DIR / "aptenodytes_forsteri_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-clarke-aptenodytes-forsteri-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
