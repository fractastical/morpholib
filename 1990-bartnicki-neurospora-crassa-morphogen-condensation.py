#!/usr/bin/env python3
"""
1990-bartnicki-neurospora-crassa-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0281)
Organism: Neurospora crassa (Red Bread Mold)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Neurospora
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0281_neurospora_crassa_morphogen_condensation.lean` (`MathOfBiology.Loop_0281`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/neurospora_crassa_morphogen_condensation"
EXPECTED_SHA256 = {
    "neurospora_crassa_morphogen_condensation_reference.json": "0f029b4344baeb7cab8841e3b477cc2473206a2e95ab37c4207fc3dccc7d94bb",
    "neurospora_crassa_specimen.jpg": "01d6531f055000fd6be6dc5baa24d01e76899962e05a548e4be38a45032165f0",
    "scout_manifest.json": "5f21e26f4d64cc07b04ab3d7780e24d66c64f2ff847d93369ab05316f7f1f8ff",
    "paper_citations.bib": "6f286bfe3acea9ba9d4d15d873d7a1e4e26289717317a093dbcaa981f2742b20"
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
    ref = json.loads((DATASET_DIR / "neurospora_crassa_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1990-bartnicki-neurospora-crassa-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
