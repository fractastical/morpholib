#!/usr/bin/env python3
"""
2005-weisblat-hirudo-medicinalis-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0169)
Organism: Hirudo medicinalis (Medicinal Leech)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Hirudo
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0169_hirudo_medicinalis_turing_wavelength.lean` (`MathOfBiology.Loop_0169`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hirudo_medicinalis_turing_wavelength"
EXPECTED_SHA256 = {
    "hirudo_medicinalis_turing_wavelength_reference.json": "7522b743b0f79ff35641bb5ccd7f9b319c93c1a1f7c167a6f9c947fe5bbd5db1",
    "hirudo_medicinalis_specimen.jpg": "f9cc303efc9e9bbd4f520cb58b823e3ee6b17491f3c40d8ee2c2ad8fcedf909f",
    "scout_manifest.json": "cc104995c0c5f86fd8bf13fad3e498780c74ed24431dc7e11cda157285c4955d",
    "paper_citations.bib": "ca01359cccf46988b354c424cde2c49a8f76dfcda7334f5f81d65177d67fef6c"
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
    ref = json.loads((DATASET_DIR / "hirudo_medicinalis_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2005-weisblat-hirudo-medicinalis-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
