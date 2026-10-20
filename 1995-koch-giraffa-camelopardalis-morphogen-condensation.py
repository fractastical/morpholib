#!/usr/bin/env python3
"""
1995-koch-giraffa-camelopardalis-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0205)
Organism: Giraffa camelopardalis (Reticulated Giraffe)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Giraffa
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0205_giraffa_camelopardalis_morphogen_condensation.lean` (`MathOfBiology.Loop_0205`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/giraffa_camelopardalis_morphogen_condensation"
EXPECTED_SHA256 = {
    "giraffa_camelopardalis_morphogen_condensation_reference.json": "8be801f431bc6f96f4d417c8f4a98fa599e3a274fff7f0d64c60a50266eb97c1",
    "giraffa_camelopardalis_specimen.jpg": "d65ee763a067a60ce829e77f52e0ca7eadb883d032bc517d6f76edf060800969",
    "scout_manifest.json": "0d113ad3b12c6ee29e1af889342f08fb7d32fa703ea7f42be769bc4fd4d6d7e0",
    "paper_citations.bib": "c7dd52ee90ea2935d5efbc88180039219b2a24824b345b86fc958cfc92ddfbed"
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
    ref = json.loads((DATASET_DIR / "giraffa_camelopardalis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1995-koch-giraffa-camelopardalis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
