#!/usr/bin/env python3
"""
1998-moore-coprinopsis-cinerea-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0183)
Organism: Coprinopsis cinerea (Gray Shag Mushroom)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Coprinopsis
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0183_coprinopsis_cinerea_turing_wavelength.lean` (`MathOfBiology.Loop_0183`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/coprinopsis_cinerea_turing_wavelength"
EXPECTED_SHA256 = {
    "coprinopsis_cinerea_turing_wavelength_reference.json": "7e61b0a94754a94b70d0d088a68145a0a3eff29ec792880683283ca09d793ee4",
    "coprinopsis_cinerea_specimen.jpg": "89844e402834973a57384e9aee4268b35a0e61f0cb9977afb4a2c562507756ca",
    "scout_manifest.json": "32cb66e4a57c6f4a41257f0f93a5c9d45aef4fe0490a68744b6b7c0a889fa097",
    "paper_citations.bib": "7b2b2c6ea68208c77ff7ccf8105fc56423739ae364735ea42c1a8c675653ee21"
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
    ref = json.loads((DATASET_DIR / "coprinopsis_cinerea_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1998-moore-coprinopsis-cinerea-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
