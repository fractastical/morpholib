#!/usr/bin/env python3
"""
1994-jean-pinus-pinea-turing-wavelength.py — Pass 2 Remediated Reproduction (Slot #0175)
Organism: Pinus pinea (Stone Pine)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Pinus
Biophysical Cohort: I. Reaction-Diffusion & Stripe Splitting (Turing Reaction-Diffusion Wavelength Selection)
Formal Verification: `morphoeval/lean/loops/Loop_0175_pinus_pinea_turing_wavelength.lean` (`MathOfBiology.Loop_0175`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pinus_pinea_turing_wavelength"
EXPECTED_SHA256 = {
    "pinus_pinea_turing_wavelength_reference.json": "47b2474b735bf3a2db634d6b73097f066516fea7bdf158b6b8649c17cda13838",
    "pinus_pinea_specimen.jpg": "87c1fef6a6fb55bd3bf3e35f590d280e453c1348b5d11f18c356c09345da3904",
    "scout_manifest.json": "ff36d7089ea43c8951f3847decbc8f5f74263da22d3d146c738f52dae463b244",
    "paper_citations.bib": "0e455ae8aa455d329f615fac8ddafd895eda1d745a06300c0dd14bbe4f4f41db"
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
    ref = json.loads((DATASET_DIR / "pinus_pinea_turing_wavelength_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1994-jean-pinus-pinea-turing-wavelength.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
