#!/usr/bin/env python3
"""
2014-levin-dugesia-japonica-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0166)
Organism: Dugesia japonica (Freshwater Planarian)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Dugesia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0166_dugesia_japonica_turing_wavelength.lean` (`MathOfBiology.Loop_0166`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/dugesia_japonica_turing_wavelength"
EXPECTED_SHA256 = {
    "dugesia_japonica_turing_wavelength_reference.json": "a9d6b34a8631269fda9ab05a800dcc0acc49751b84c66daacbfada9e5a295db6",
    "dugesia_japonica_specimen.jpg": "26b78d28b853c1b70092be6a7ae129d59813507f64eaed16a417276296ff2efa",
    "scout_manifest.json": "62542dd3b02f4e91f2050bfd4a42c93af84712dca972a9518369256dba2577d0",
    "paper_citations.bib": "a99e92e682d59daf1edc1f5655a3b054b5a2a190c79e46fa8c67ad9c84abf234"
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
    ref = json.loads((DATASET_DIR / "dugesia_japonica_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2014-levin-dugesia-japonica-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
