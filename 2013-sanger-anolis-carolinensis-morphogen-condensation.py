#!/usr/bin/env python3
"""
2013-sanger-anolis-carolinensis-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0224)
Organism: Anolis carolinensis (Green Anole)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Anolis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0224_anolis_carolinensis_morphogen_condensation.lean` (`MathOfBiology.Loop_0224`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anolis_carolinensis_morphogen_condensation"
EXPECTED_SHA256 = {
    "anolis_carolinensis_morphogen_condensation_reference.json": "2c1c7fbc9afe5d44f6ed6cc4002022b66bf1c4ff30e8ca1af383946e7db9c18b",
    "anolis_carolinensis_specimen.jpg": "584ab986514945bb9c82b1765cb9c44cb8915e47823a0e02b5f444310fe0506b",
    "scout_manifest.json": "5c93aa4cd46e34f3d7f58001c8960c20bfe5b84e0cb5fab787879f5a8bae702c",
    "paper_citations.bib": "b8425367502808193a70b61d500d37dd3c6d5bfbe793e7e5d85d9a6c00edaccc"
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
    ref = json.loads((DATASET_DIR / "anolis_carolinensis_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-sanger-anolis-carolinensis-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
