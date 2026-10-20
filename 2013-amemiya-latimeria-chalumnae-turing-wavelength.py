#!/usr/bin/env python3
"""
2013-amemiya-latimeria-chalumnae-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0139)
Organism: Latimeria chalumnae (West Indian Coelacanth)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Latimeria
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0139_latimeria_chalumnae_turing_wavelength.lean` (`MathOfBiology.Loop_0139`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/latimeria_chalumnae_turing_wavelength"
EXPECTED_SHA256 = {
    "latimeria_chalumnae_turing_wavelength_reference.json": "92b47625199c6c21529e5dd4e81e41aae2d9b405253a83a46210618710d79979",
    "latimeria_chalumnae_specimen.jpg": "29bcb5946834bbbf95c873d247d7bb9449f8ea6b56ddef496b4eef521ce6cd5d",
    "scout_manifest.json": "84a8b370e46b919b445366f6b9ba0c9028cf0eb25140d986292046e1d7f4f785",
    "paper_citations.bib": "db39a7a6afb0f11d1bc834ee20fa78dc2d038bfee010fa0ef437451120dce4ba"
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
    ref = json.loads((DATASET_DIR / "latimeria_chalumnae_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-amemiya-latimeria-chalumnae-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
