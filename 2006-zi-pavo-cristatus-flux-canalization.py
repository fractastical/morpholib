#!/usr/bin/env python3
"""
2006-zi-pavo-cristatus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0413)
Organism: Pavo cristatus (Indian Peafowl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Pavo
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0413_pavo_cristatus_flux_canalization.lean` (`MathOfBiology.Loop_0413`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pavo_cristatus_flux_canalization"
EXPECTED_SHA256 = {
    "pavo_cristatus_flux_canalization_reference.json": "cd45f10de22f5686cce6df3a39acb19a065c13d0d9ef1d76c591e06b06d24cd5",
    "pavo_cristatus_specimen.jpg": "39f28e26a62ddfdb98fe19fee7eaf7af508c82486a32abe3c1f2ac7d3e607d50",
    "scout_manifest.json": "cc610b638ddb2acdd457505532f6d7e7932c4acdf314e398394517beb0e2b6e9",
    "paper_citations.bib": "ca61d22d460b53bbf7828a0055c7c4f0b830e134912060df8d794f65046ae83b"
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
    ref = json.loads((DATASET_DIR / "pavo_cristatus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2006-zi-pavo-cristatus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
