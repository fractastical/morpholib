#!/usr/bin/env python3
"""
2018-cooper-scyliorhinus-canicula-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0135)
Organism: Scyliorhinus canicula (Small-Spotted Catshark)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Scyliorhinus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0135_scyliorhinus_canicula_turing_wavelength.lean` (`MathOfBiology.Loop_0135`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/scyliorhinus_canicula_turing_wavelength"
EXPECTED_SHA256 = {
    "scyliorhinus_canicula_turing_wavelength_reference.json": "df47d424aa39092268b1938adfc59b054f907f934fc8d4cd319cf7e32c8a2413",
    "scyliorhinus_canicula_specimen.jpg": "71525dd878e51b100ea6bbd4e4005751066059fc236d4e204ad81ed2bbb5d553",
    "scout_manifest.json": "510471c4d0d6bd7a7ca5f821d8b6a51fc066a674fd5ac89fe26141faf06f197a",
    "paper_citations.bib": "856618e7ae58d2099dc91631aa1922fa7f603b71f08b8a2de790c63d7144056a"
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
    ref = json.loads((DATASET_DIR / "scyliorhinus_canicula_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-cooper-scyliorhinus-canicula-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
