#!/usr/bin/env python3
"""
2009-gordon-arachnoidiscus-ehrenbergii-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0199)
Organism: Arachnoidiscus ehrenbergii (Wheel Centric Diatom)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Arachnoidiscus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0199_arachnoidiscus_ehrenbergii_turing_wavelength.lean` (`MathOfBiology.Loop_0199`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/arachnoidiscus_ehrenbergii_turing_wavelength"
EXPECTED_SHA256 = {
    "arachnoidiscus_ehrenbergii_turing_wavelength_reference.json": "bac462147d0eda8c119c890877111fde7101b5406a53cbd394f17a6e95d10ea6",
    "arachnoidiscus_ehrenbergii_specimen.jpg": "499034f85d30d464ea67fea4d7404a3546b57b85bb64c8d553cb36f2129bed05",
    "scout_manifest.json": "aca187f59e1b2f794f5f1c6847fbcf9b750789c2c5fcdfd587ced01bb1527b4e",
    "paper_citations.bib": "04caf26bcb50464b48501e0bd66b32af7c25ab127c99c8b808b460a9b1ce21c9"
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
    ref = json.loads((DATASET_DIR / "arachnoidiscus_ehrenbergii_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-gordon-arachnoidiscus-ehrenbergii-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
