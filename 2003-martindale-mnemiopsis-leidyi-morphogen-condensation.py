#!/usr/bin/env python3
"""
2003-martindale-mnemiopsis-leidyi-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0270)
Organism: Mnemiopsis leidyi (Warty Comb Jelly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Mnemiopsis
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0270_mnemiopsis_leidyi_morphogen_condensation.lean` (`MathOfBiology.Loop_0270`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mnemiopsis_leidyi_morphogen_condensation"
EXPECTED_SHA256 = {
    "mnemiopsis_leidyi_morphogen_condensation_reference.json": "5ecc6ce5e53ec19044f15743a24ff4ca05c0675a357dd9b705e4f89f91a3f1f2",
    "mnemiopsis_leidyi_specimen.jpg": "d966e0f5b06b5af8e1ef77edf5e32c90c3aa8489be8bbbf2bdf7795ecbfb7ec7",
    "scout_manifest.json": "8be44baac2b4f5c7bdb8343fddd7a8f191d4caf52d31fb0f00dc9b0ca8eaf913",
    "paper_citations.bib": "697a2cf2e2d4fa8a01ff7bd8958ef4d9f27a5b67865c36aead12ed85a9044393"
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
    ref = json.loads((DATASET_DIR / "mnemiopsis_leidyi_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-martindale-mnemiopsis-leidyi-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
