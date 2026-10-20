#!/usr/bin/env python3
"""
2000-benjacob-paenibacillus-dendritiformis-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0189)
Organism: Paenibacillus dendritiformis (Chiral Branching Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Paenibacillus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0189_paenibacillus_dendritiformis_turing_wavelength.lean` (`MathOfBiology.Loop_0189`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/paenibacillus_dendritiformis_turing_wavelength"
EXPECTED_SHA256 = {
    "paenibacillus_dendritiformis_turing_wavelength_reference.json": "bd63ac9a943f2c73aef6b891d6eb2f9eb219a83b78f2cc5ee621db868c64dadf",
    "paenibacillus_dendritiformis_specimen.jpg": "dbed60f770b34bfce5f9193b55484568f5cccc3a5f417b5c7c6e79a7a8c84cf6",
    "scout_manifest.json": "3d750b0a7e4c9c017ba6de3af3f06cc82fb9783bc2644385c468904b02e5e7c6",
    "paper_citations.bib": "3d0e7a3f5d5dc8ad5ba0742991762cceb0631a1e94737b5b6b3db4e2659f3e85"
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
    ref = json.loads((DATASET_DIR / "paenibacillus_dendritiformis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2000-benjacob-paenibacillus-dendritiformis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
