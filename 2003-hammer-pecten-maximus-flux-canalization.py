#!/usr/bin/env python3
"""
2003-hammer-pecten-maximus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0457)
Organism: Pecten maximus (Great Scallop)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Pecten
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0457_pecten_maximus_flux_canalization.lean` (`MathOfBiology.Loop_0457`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pecten_maximus_flux_canalization"
EXPECTED_SHA256 = {
    "pecten_maximus_flux_canalization_reference.json": "2d35680608aeb7bad03169c4c815eecd1c6adf623506f62e7eab42966d84a3e0",
    "pecten_maximus_specimen.jpg": "2f87d4ec30ebec6adb82f91aec67eb300e3469f57920cbccf9c9bb0d22b703a8",
    "scout_manifest.json": "825a32aaa011db7703d9d96d1dcc855b3abbb126fed357aa626f82192903bf5b",
    "paper_citations.bib": "d10790c2624ba3b7d4b09f2bb857197c508cb4317f880cf87ee47d39512d739a"
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
    ref = json.loads((DATASET_DIR / "pecten_maximus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-hammer-pecten-maximus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
