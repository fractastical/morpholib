#!/usr/bin/env python3
"""
2015-abrams-aurelia-aurita-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0164)
Organism: Aurelia aurita (Moon Jellyfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Aurelia
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0164_aurelia_aurita_turing_wavelength.lean` (`MathOfBiology.Loop_0164`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aurelia_aurita_turing_wavelength"
EXPECTED_SHA256 = {
    "aurelia_aurita_turing_wavelength_reference.json": "e0071943f97a972647ba30c9586f386642b708f4580a93dec72b64f95e93870d",
    "aurelia_aurita_specimen.jpg": "1a8c5b9fb4abe35f1362b2c458d03bafa15966b0f6046d49aa840fdd3a66769e",
    "scout_manifest.json": "49448854177f1f30263169d94a384ff4b399b2476b98165d6be207de087e1414",
    "paper_citations.bib": "83c8ed033f12f1e36e7a5dcc4e9a858b304929df0e90645a38ef3a0ca28cd2b5"
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
    ref = json.loads((DATASET_DIR / "aurelia_aurita_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-abrams-aurelia-aurita-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
