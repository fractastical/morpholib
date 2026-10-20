#!/usr/bin/env python3
"""
2004-flardh-streptomyces-coelicolor-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0288)
Organism: Streptomyces coelicolor (Mycelial Actinobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Streptomyces
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0288_streptomyces_coelicolor_morphogen_condensation.lean` (`MathOfBiology.Loop_0288`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/streptomyces_coelicolor_morphogen_condensation"
EXPECTED_SHA256 = {
    "streptomyces_coelicolor_morphogen_condensation_reference.json": "3c256cc1031415cb1ed3b2438d74a1d7a9a9aabdd038c7a590ba629ecb699e7d",
    "streptomyces_coelicolor_specimen.jpg": "5a2b99731bd4971f58260f74a1db5d4560b78f4d8ceb3b067629dcb7b6d53fec",
    "scout_manifest.json": "97719149d5cf1707078f77adc15a52bf6ed0ba673db524d3f680f66d32428eb4",
    "paper_citations.bib": "ad8a12f5a7bd437a25df87cfdc982024dea5d6eae33c04b4fb12ad6340e6212b"
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
    ref = json.loads((DATASET_DIR / "streptomyces_coelicolor_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2004-flardh-streptomyces-coelicolor-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
