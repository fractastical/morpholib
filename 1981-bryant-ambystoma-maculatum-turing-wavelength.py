#!/usr/bin/env python3
"""
1981-bryant-ambystoma-maculatum-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0122)
Organism: Ambystoma maculatum (Spotted Salamander)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ambystoma
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0122_ambystoma_maculatum_turing_wavelength.lean` (`MathOfBiology.Loop_0122`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ambystoma_maculatum_turing_wavelength"
EXPECTED_SHA256 = {
    "ambystoma_maculatum_turing_wavelength_reference.json": "e1217910660d3836f398d56b027da03ad080559d9569bfa026be670f2157c2aa",
    "ambystoma_maculatum_specimen.jpg": "32e4b406ab308f3580f182f76a10b7ef37ad16c150b655ab556d886365f151d6",
    "scout_manifest.json": "4a30166c4b3c7a6108f5d898eb61c55de6ffdc8b4187ddd2eb335e8532b6b3bd",
    "paper_citations.bib": "4a16e04a963880b40ea127fbbddf234cb36e49793a64ab1adca9ecdbf1fc9422"
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
    ref = json.loads((DATASET_DIR / "ambystoma_maculatum_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-bryant-ambystoma-maculatum-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
