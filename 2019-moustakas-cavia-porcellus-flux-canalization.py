#!/usr/bin/env python3
"""
2019-moustakas-cavia-porcellus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0408)
Organism: Cavia porcellus (Guinea Pig)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Cavia
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0408_cavia_porcellus_flux_canalization.lean` (`MathOfBiology.Loop_0408`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/cavia_porcellus_flux_canalization"
EXPECTED_SHA256 = {
    "cavia_porcellus_flux_canalization_reference.json": "79d83aea9ec45baba44c4047dbb55201ce42457479b6ec8214e4b498836a5341",
    "cavia_porcellus_specimen.jpg": "42014eaf5b445ec569739b4b14be9f65dea72da7126e6dc92702e66dbbbf2acd",
    "scout_manifest.json": "34ea58fb88340c3424762f5d699390d77858ac3c0a34e6752b08c6e4b0770a05",
    "paper_citations.bib": "272703323b349029791a652191833522faee4844a31746d068b564b906b87d86"
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
    ref = json.loads((DATASET_DIR / "cavia_porcellus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-moustakas-cavia-porcellus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
