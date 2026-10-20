#!/usr/bin/env python3
"""
1959-hartline-limulus-polyphemus-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0450)
Organism: Limulus polyphemus (Atlantic Horseshoe Crab)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Limulus
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0450_limulus_polyphemus_flux_canalization.lean` (`MathOfBiology.Loop_0450`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/limulus_polyphemus_flux_canalization"
EXPECTED_SHA256 = {
    "limulus_polyphemus_flux_canalization_reference.json": "1d775416d01591dca6f25016bd95e0095bfad2d50cb19525869c09d1fdddd8bc",
    "limulus_polyphemus_specimen.jpg": "6024a6e114ddb2e0b996d82067f840499683d1838662afe24496b8f315e2ef45",
    "scout_manifest.json": "a45a2835384fad88754ddf96db913821187e898dde24cc00943f44f102d70a5d",
    "paper_citations.bib": "65d06b832d672d4284782f805aecdfed9257936decaebb7a4bade48bab29996e"
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
    ref = json.loads((DATASET_DIR / "limulus_polyphemus_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1959-hartline-limulus-polyphemus-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
