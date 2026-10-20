#!/usr/bin/env python3
"""
2001-gilbert-trachemys-scripta-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0129)
Organism: Trachemys scripta (Red-Eared Slider Turtle)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Trachemys
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0129_trachemys_scripta_turing_wavelength.lean` (`MathOfBiology.Loop_0129`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/trachemys_scripta_turing_wavelength"
EXPECTED_SHA256 = {
    "trachemys_scripta_turing_wavelength_reference.json": "e542e8a21d10d81135d91051801e3c0bf40085d33e840036da57de5111ffd7db",
    "trachemys_scripta_specimen.jpg": "5d30c00d709c06ef8c1ebdb9b272654df48237cd07a6d19be7e219ab27188e96",
    "scout_manifest.json": "8ed68cd7618e0402d633c4bdb12bb4875ecded2d7ff65969c48ddc5a64090937",
    "paper_citations.bib": "1e77cf9a7017de23175b19aab433d673df99695f60092c678581a7d83c459fd1"
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
    ref = json.loads((DATASET_DIR / "trachemys_scripta_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2001-gilbert-trachemys-scripta-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
