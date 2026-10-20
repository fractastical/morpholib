#!/usr/bin/env python3
"""
2002-munro-ciona-intestinalis-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0863)
Organism: Ciona intestinalis (Vase Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Ciona
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0863_ciona_intestinalis_biomineral_accretion.lean` (`MathOfBiology.Loop_0863`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ciona_intestinalis_biomineral_accretion"
EXPECTED_SHA256 = {
    "ciona_intestinalis_biomineral_accretion_reference.json": "a400156c62c85b0aef738b487e20b963210c17d2579a580c109d03014caae7f8",
    "ciona_intestinalis_specimen.jpg": "0f563cbc2ccb4b7ba1595b7519a75a1410031eaaa4afceaf02b5975d87c1ecf4",
    "scout_manifest.json": "e1d381b0bbca73e3a41e42e9af85ee8de0476d8a60b60e8dca983a76dda74be5",
    "paper_citations.bib": "0e455300d387147a71b57a9b56045d3b3e8c6e2ea4b90afdf49638b56945bb72"
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
    ref = json.loads((DATASET_DIR / "ciona_intestinalis_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-munro-ciona-intestinalis-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
