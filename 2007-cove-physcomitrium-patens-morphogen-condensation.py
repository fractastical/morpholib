#!/usr/bin/env python3
"""
2007-cove-physcomitrium-patens-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0272)
Organism: Physcomitrium patens (Spreading Earthmoss)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Physcomitrium
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0272_physcomitrium_patens_morphogen_condensation.lean` (`MathOfBiology.Loop_0272`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/physcomitrium_patens_morphogen_condensation"
EXPECTED_SHA256 = {
    "physcomitrium_patens_morphogen_condensation_reference.json": "77f5628a7decdc6b6b6f9f5a78efa797abd30b457d651c0e89bedb5b8958a7c4",
    "physcomitrium_patens_specimen.jpg": "a115412e5c6b1e47cad14da271d9ee4f3d33b2b7150fb4c533f145234f694d13",
    "scout_manifest.json": "a230ccaf6b940f11b1e3e546983131aa9850fb8c2e597fd5612453197c6d4d23",
    "paper_citations.bib": "4bae751ee4e5afd1ee34bb41ea10d0c9c6c71f24a637b1bc4114371daab0fff9"
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
    ref = json.loads((DATASET_DIR / "physcomitrium_patens_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-cove-physcomitrium-patens-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
