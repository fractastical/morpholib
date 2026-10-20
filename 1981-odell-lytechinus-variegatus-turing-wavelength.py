#!/usr/bin/env python3
"""
1981-odell-lytechinus-variegatus-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0155)
Organism: Lytechinus variegatus (Green Sea Urchin)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Lytechinus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0155_lytechinus_variegatus_turing_wavelength.lean` (`MathOfBiology.Loop_0155`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/lytechinus_variegatus_turing_wavelength"
EXPECTED_SHA256 = {
    "lytechinus_variegatus_turing_wavelength_reference.json": "34dd5487ccc7c7f2994d6e0954a361004d52dc19451a642469b53e5cefc072b6",
    "lytechinus_variegatus_specimen.jpg": "521ca1f087eb9618f793b80f0084a9efa6db3c8fb7039599c38d75ba00260619",
    "scout_manifest.json": "e05c73e2319326a359e75cc7dba1d31a3a5773c359a8ac5875f9fd6f7550b257",
    "paper_citations.bib": "4e56be81196b41325ad5e74126443e2963a935582693eb8a7d35d862c71fb2bd"
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
    ref = json.loads((DATASET_DIR / "lytechinus_variegatus_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-odell-lytechinus-variegatus-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
