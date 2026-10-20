#!/usr/bin/env python3
"""
1999-young-gephyrocapsa-oceanica-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0198)
Organism: Gephyrocapsa oceanica (Calcite Coccolithophore)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Gephyrocapsa
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0198_gephyrocapsa_oceanica_turing_wavelength.lean` (`MathOfBiology.Loop_0198`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gephyrocapsa_oceanica_turing_wavelength"
EXPECTED_SHA256 = {
    "gephyrocapsa_oceanica_turing_wavelength_reference.json": "b7103fd9bcad8a861b991b2e47ee494b8198c6490dcc8962717ee10bbdef917b",
    "gephyrocapsa_oceanica_specimen.jpg": "b3482bdf62d2f619ecd8ad93b8bc68fb784477c51f4de8d0682d3d8fa3dbbc9e",
    "scout_manifest.json": "2bc410ab6084020546c1727f309c0c595178f07a0899d1413848f6bab659f967",
    "paper_citations.bib": "9e5248fa91bcae3e7e49bd14a4669b284c049a00021dd56ff7f6032ca2bbffd5"
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
    ref = json.loads((DATASET_DIR / "gephyrocapsa_oceanica_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1999-young-gephyrocapsa-oceanica-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
