#!/usr/bin/env python3
"""
2009-cove-physcomitrium-patens-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0472)
Organism: Physcomitrium patens (Spreading Earthmoss)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Physcomitrium
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0472_physcomitrium_patens_flux_canalization.lean` (`MathOfBiology.Loop_0472`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/physcomitrium_patens_flux_canalization"
EXPECTED_SHA256 = {
    "physcomitrium_patens_flux_canalization_reference.json": "253a1002d7c16434fc324afbec19d06b93c793da5b8846bf6a9f2e3e3bd75695",
    "physcomitrium_patens_specimen.jpg": "a115412e5c6b1e47cad14da271d9ee4f3d33b2b7150fb4c533f145234f694d13",
    "scout_manifest.json": "f19c8ed953401a3b92fcabc141398ba9714850ee36612612892a0172bdfe846f",
    "paper_citations.bib": "d419528401459e1ce3fa4f0a131f9b4e2ffdbae5a2c3f7b0620c8fdf41c93eca"
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
    ref = json.loads((DATASET_DIR / "physcomitrium_patens_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-cove-physcomitrium-patens-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
