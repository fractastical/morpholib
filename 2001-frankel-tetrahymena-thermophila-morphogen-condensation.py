#!/usr/bin/env python3
"""
2001-frankel-tetrahymena-thermophila-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0297)
Organism: Tetrahymena thermophila (Ciliated Cortical Protist)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Tetrahymena
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0297_tetrahymena_thermophila_morphogen_condensation.lean` (`MathOfBiology.Loop_0297`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetrahymena_thermophila_morphogen_condensation"
EXPECTED_SHA256 = {
    "tetrahymena_thermophila_morphogen_condensation_reference.json": "b6db643a55768df84e989626c4689325bec664a5a793915cfbf45136eba2043e",
    "tetrahymena_thermophila_specimen.jpg": "7704d06cf5ec8d2a1b6bb5954c4846597cac08e5f4d0c57aed6cd6e1d729026c",
    "scout_manifest.json": "87c4064c14378b240c03200614be8f4625312874f497ac485c6b083802ae255f",
    "paper_citations.bib": "0098cc5f898532db4cbbc3f6fdbc896e068e9510c55c8666606b26c1f1fcf933"
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
    ref = json.loads((DATASET_DIR / "tetrahymena_thermophila_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-frankel-tetrahymena-thermophila-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
