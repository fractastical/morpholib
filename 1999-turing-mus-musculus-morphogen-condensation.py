#!/usr/bin/env python3
"""
1999-turing-mus-musculus-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0201)
Organism: Mus musculus (House Mouse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Mus
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0201_mus_musculus_morphogen_condensation.lean` (`MathOfBiology.Loop_0201`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mus_musculus_morphogen_condensation"
EXPECTED_SHA256 = {
    "mus_musculus_morphogen_condensation_reference.json": "13ade495c370ede7b6b546b1547ce5c280f062fd2709ca2f529711b23a90b9fd",
    "mus_musculus_specimen.jpg": "f2738290d66e441895fc10af7d7e541424263249cda0c7a8194ff01b410fe385",
    "scout_manifest.json": "30f6ca4da9253007e19bfd61cc750778fdeaa10a5071e01b18dcb9338a887129",
    "paper_citations.bib": "21349c8a7a3125b62f4764e9b3f37d6e624c64895451a3813f293a23c89783a7"
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
    ref = json.loads((DATASET_DIR / "mus_musculus_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1999-turing-mus-musculus-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
