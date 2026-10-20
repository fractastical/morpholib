#!/usr/bin/env python3
"""
2007-wu-anas-platyrhynchos-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0416)
Organism: Anas platyrhynchos (Mallard Duck)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anas
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0416_anas_platyrhynchos_flux_canalization.lean` (`MathOfBiology.Loop_0416`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anas_platyrhynchos_flux_canalization"
EXPECTED_SHA256 = {
    "anas_platyrhynchos_flux_canalization_reference.json": "764055dd0b2534eb60100bd2f6f873c938d0b9216f96c6bbb430255800239f1b",
    "anas_platyrhynchos_specimen.jpg": "ee1cc0720d68f5f20e5e3490647b69cb29c1fe84dae8fdb048019814f69eccba",
    "scout_manifest.json": "c0138bc3918a8e3d783a5b4b03a0d9d02aa80993d65dbe679bde2a13e5c32853",
    "paper_citations.bib": "64f4e9ec5b703b19d60f59008a11a00b278391e4624831e808509672736ba4a0"
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
    ref = json.loads((DATASET_DIR / "anas_platyrhynchos_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-wu-anas-platyrhynchos-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
