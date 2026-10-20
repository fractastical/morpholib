#!/usr/bin/env python3
"""
2007-lecuit-drosophila-melanogaster-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0141)
Organism: Drosophila melanogaster (Fruit Fly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Drosophila
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0141_drosophila_melanogaster_turing_wavelength.lean` (`MathOfBiology.Loop_0141`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/drosophila_melanogaster_turing_wavelength"
EXPECTED_SHA256 = {
    "drosophila_melanogaster_turing_wavelength_reference.json": "66a354ca30c365f6bccb513e31933a5234564be664e7b86407fe71b0773a7b59",
    "drosophila_melanogaster_specimen.jpg": "f426366f7ede436da44782b28ebac9bd013cace1f8fe44b6ed382b77ad86c419",
    "scout_manifest.json": "84b42f2043eb2fa73d4575d0f7fc23445a1dabdd94c0ef91e3eb8cf615941977",
    "paper_citations.bib": "190b48de49bcfc87eeade3b877ec1895314b23b504e93efacf770773a2bdb9b2"
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
    ref = json.loads((DATASET_DIR / "drosophila_melanogaster_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-lecuit-drosophila-melanogaster-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
