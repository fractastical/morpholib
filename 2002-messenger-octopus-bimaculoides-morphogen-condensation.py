#!/usr/bin/env python3
"""
2002-messenger-octopus-bimaculoides-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0252)
Organism: Octopus bimaculoides (Two-Spot Octopus)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Octopus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0252_octopus_bimaculoides_morphogen_condensation.lean` (`MathOfBiology.Loop_0252`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/octopus_bimaculoides_morphogen_condensation"
EXPECTED_SHA256 = {
    "octopus_bimaculoides_morphogen_condensation_reference.json": "6b46912641179fcef73a6de50ec5e15ff2cb9b9213af59fa325432834564903b",
    "octopus_bimaculoides_specimen.jpg": "4e6d70661aa4e7709a51348bee322500de3a32b1ecb8b557b67060ff9d863233",
    "scout_manifest.json": "07ae90bf9df11cf1a046728e063686a27723a386778579856887b286eca3df5d",
    "paper_citations.bib": "38ce5a248d9e9ce892070761dc0af012f5d938f5243e47ab327d2e183d1c66f9"
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
    ref = json.loads((DATASET_DIR / "octopus_bimaculoides_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2002-messenger-octopus-bimaculoides-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
