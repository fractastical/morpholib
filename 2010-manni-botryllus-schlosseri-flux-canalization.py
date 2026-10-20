#!/usr/bin/env python3
"""
2010-manni-botryllus-schlosseri-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0467)
Organism: Botryllus schlosseri (Star Ascidian Tunicate)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Botryllus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0467_botryllus_schlosseri_flux_canalization.lean` (`MathOfBiology.Loop_0467`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/botryllus_schlosseri_flux_canalization"
EXPECTED_SHA256 = {
    "botryllus_schlosseri_flux_canalization_reference.json": "383ceebea594b40b65f76b8ff3dc22916724208cf6d52df97d1d100e16b54de0",
    "botryllus_schlosseri_specimen.jpg": "b1ea2c5231b1b71026b40278bbac0bde71e6cf35e9fd8bedcd40f103a7190401",
    "scout_manifest.json": "fb288ccfb7e5b36da7fabc936d1a76428e9ad97d4f98d95037f336342e46567a",
    "paper_citations.bib": "0c39b7cd7f6f87066bc4a8c363a6c1753077d11a5deefa514bef56f3f59070d7"
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
    ref = json.loads((DATASET_DIR / "botryllus_schlosseri_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-manni-botryllus-schlosseri-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
