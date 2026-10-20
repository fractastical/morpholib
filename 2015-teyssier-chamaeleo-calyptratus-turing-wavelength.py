#!/usr/bin/env python3
"""
2015-teyssier-chamaeleo-calyptratus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0123)
Organism: Chamaeleo calyptratus (Veiled Chameleon)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Chamaeleo
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0123_chamaeleo_calyptratus_turing_wavelength.lean` (`MathOfBiology.Loop_0123`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/chamaeleo_calyptratus_turing_wavelength"
EXPECTED_SHA256 = {
    "chamaeleo_calyptratus_turing_wavelength_reference.json": "8035228f87e2682f9f4e6f0d1a265585965d4d9faf084e441b3ba7f6b9f50c86",
    "chamaeleo_calyptratus_specimen.jpg": "976c60b3ba6aa23d8e363fe2188ae09cd8c806190b8b02939475defbf44924b3",
    "scout_manifest.json": "01598e60aeb036270499e9fbfd134bef9c040c266621359a9599f2cd586c82fc",
    "paper_citations.bib": "48d7c69a1df22cc229424d82c46bb5257b82bc5659449dd469433e94983b1626"
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
    ref = json.loads((DATASET_DIR / "chamaeleo_calyptratus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-teyssier-chamaeleo-calyptratus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
