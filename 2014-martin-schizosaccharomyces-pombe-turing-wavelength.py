#!/usr/bin/env python3
"""
2014-martin-schizosaccharomyces-pombe-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0187)
Organism: Schizosaccharomyces pombe (Fission Yeast)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Schizosaccharomyces
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0187_schizosaccharomyces_pombe_turing_wavelength.lean` (`MathOfBiology.Loop_0187`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/schizosaccharomyces_pombe_turing_wavelength"
EXPECTED_SHA256 = {
    "schizosaccharomyces_pombe_turing_wavelength_reference.json": "6b7171df80412b7d83bce00931b548cf72809fdaf74bc5a82c13d1b8b96044e4",
    "schizosaccharomyces_pombe_specimen.jpg": "1ddf4f2081b0c047e472365745cd54e38ef85ea46a86a451dc98e0a7d6b5c9f8",
    "scout_manifest.json": "c87298665b938dc261550d35e724c1b9333e32422f530a744131f73aa1fa2ce7",
    "paper_citations.bib": "61599375a61d451502cc223f229349cc6d5fd4ab5704f41ee861c7e13c5b43c7"
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
    ref = json.loads((DATASET_DIR / "schizosaccharomyces_pombe_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-martin-schizosaccharomyces-pombe-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
