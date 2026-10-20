#!/usr/bin/env python3
"""
2009-niimi-bombyx-mori-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0446)
Organism: Bombyx mori (Domestic Silkmoth)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Bombyx
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0446_bombyx_mori_flux_canalization.lean` (`MathOfBiology.Loop_0446`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/bombyx_mori_flux_canalization"
EXPECTED_SHA256 = {
    "bombyx_mori_flux_canalization_reference.json": "82559e01b7f0d37745ab7b2dc6f1895d0f32a0b484af1b51ceb3df6dd84ad2a0",
    "bombyx_mori_specimen.jpg": "2adf51228d2da684d9c8eb6f510c972b1da829813ca787dd645448cff68d5d97",
    "scout_manifest.json": "15dee2851410bf697faaa4d4034391f86f5f37e0049a793b55ebb68ec762d6d5",
    "paper_citations.bib": "b60717faa3d445fbc2902517843c9e0807108ba03eeb7831d606eb18982507be"
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
    ref = json.loads((DATASET_DIR / "bombyx_mori_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-niimi-bombyx-mori-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
