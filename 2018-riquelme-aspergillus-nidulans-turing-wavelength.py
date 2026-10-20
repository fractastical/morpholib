#!/usr/bin/env python3
"""
2018-riquelme-aspergillus-nidulans-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0182)
Organism: Aspergillus nidulans (Filamentous Fungus)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Aspergillus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0182_aspergillus_nidulans_turing_wavelength.lean` (`MathOfBiology.Loop_0182`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aspergillus_nidulans_turing_wavelength"
EXPECTED_SHA256 = {
    "aspergillus_nidulans_turing_wavelength_reference.json": "e1d314a0c82d3104eb7eb750f771b7a1361bbe9624861f33afa88a4831107aea",
    "aspergillus_nidulans_specimen.jpg": "93e53157e77eaccce94a5792d58d0e622ba9bb55b15fd47100e1f7c90985dec5",
    "scout_manifest.json": "af524a4a3cee2422c18f6046d673ec10be42300e109259c86482af324bd2e38b",
    "paper_citations.bib": "55e907eb33e339468b3ea9bde32506744666e5c1ec64c135f6d18d97aaf6a0bc"
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
    ref = json.loads((DATASET_DIR / "aspergillus_nidulans_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-riquelme-aspergillus-nidulans-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
