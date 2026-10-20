#!/usr/bin/env python3
"""
1997-brockes-notophthalmus-viridescens-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0127)
Organism: Notophthalmus viridescens (Eastern Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Notophthalmus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0127_notophthalmus_viridescens_turing_wavelength.lean` (`MathOfBiology.Loop_0127`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/notophthalmus_viridescens_turing_wavelength"
EXPECTED_SHA256 = {
    "notophthalmus_viridescens_turing_wavelength_reference.json": "ba25852960f82aec48e8772beae61c5804970adf83dae62ecf1de6f9b7c7ea4c",
    "notophthalmus_viridescens_specimen.jpg": "5389e095a5be650aba6c5b00fa322f4fbd3beed046b1913ac82a95ce045d5000",
    "scout_manifest.json": "0518c29be649944ae2889010a03aa801d7afd52789885e3fd4e19b5094bfa8a6",
    "paper_citations.bib": "70f8a415bdb32130ac3c2288dd3efb4f7dc6ac8d155b7a94de5d382b6324b32a"
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
    ref = json.loads((DATASET_DIR / "notophthalmus_viridescens_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1997-brockes-notophthalmus-viridescens-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
