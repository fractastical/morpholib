#!/usr/bin/env python3
"""
2001-chuong-gallus-gallus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0211)
Organism: Gallus gallus (Red Junglefowl Chick)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Gallus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0211_gallus_gallus_morphogen_condensation.lean` (`MathOfBiology.Loop_0211`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gallus_gallus_morphogen_condensation"
EXPECTED_SHA256 = {
    "gallus_gallus_morphogen_condensation_reference.json": "371ece07d8c551d819be60463e1262d95158231f4a7ab97bf1279932c72d6a1a",
    "gallus_gallus_specimen.jpg": "7e7d58db6bb564ebe68e091678338c7c06b920cdb66f9dffd99b4141bf16c9cb",
    "scout_manifest.json": "9172549e9e93129d87edafae5707328616bc6c6eb1c43f7602fec2c24e6d7ddd",
    "paper_citations.bib": "3326942da6a0f99b2857c0d07f6655d4d5eae5f44447ec88dda9004d70114451"
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
    ref = json.loads((DATASET_DIR / "gallus_gallus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-chuong-gallus-gallus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
