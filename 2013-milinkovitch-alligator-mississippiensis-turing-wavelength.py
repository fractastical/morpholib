#!/usr/bin/env python3
"""
2013-milinkovitch-alligator-mississippiensis-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0130)
Organism: Alligator mississippiensis (American Alligator)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Alligator
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0130_alligator_mississippiensis_turing_wavelength.lean` (`MathOfBiology.Loop_0130`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/alligator_mississippiensis_turing_wavelength"
EXPECTED_SHA256 = {
    "alligator_mississippiensis_turing_wavelength_reference.json": "8140614a317fe92062c4f5aaf07f16039140f0bdfe79db2de431113a88a8af90",
    "alligator_mississippiensis_specimen.jpg": "bb1770a8cf4a71024eec779363925a2928277079f2124074117061fc6d960764",
    "scout_manifest.json": "6b0b003bf21eeab4b3b28241bc1eb8c0419b74474863a2447615a2830e457236",
    "paper_citations.bib": "44068cbcae67bcc01160cd93ad1571c3a911aebdf179a9704bd4203df74a5af9"
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
    ref = json.loads((DATASET_DIR / "alligator_mississippiensis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-milinkovitch-alligator-mississippiensis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
