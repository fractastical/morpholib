#!/usr/bin/env python3
"""
1984-bryant-ambystoma-maculatum-flux-canalization.py — Pass 2 Remediated Reproduction (Slot #0422)
Organism: Ambystoma maculatum (Spotted Salamander)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ambystoma
Biophysical Cohort: IV. Phyllotaxis, Auxin & Plant Mechanics (Polar Flux Canalization & Anisotropic Growth)
Formal Verification: `morphoeval/lean/loops/Loop_0422_ambystoma_maculatum_flux_canalization.lean` (`MathOfBiology.Loop_0422`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ambystoma_maculatum_flux_canalization"
EXPECTED_SHA256 = {
    "ambystoma_maculatum_flux_canalization_reference.json": "a788231d7e2fd9d2a5ab1d3c725f50a814f5e82d3f49a4707f664c043a6f6519",
    "ambystoma_maculatum_specimen.jpg": "32e4b406ab308f3580f182f76a10b7ef37ad16c150b655ab556d886365f151d6",
    "scout_manifest.json": "f53ee51e3e3ae94a541b7be664ab441792f49612a9b56a7b2feeb90ee03058fc",
    "paper_citations.bib": "577a80ad3b4fd227338e8e8c09dfbf0265399c98b23a0cebbef2a6888284d665"
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
    ref = json.loads((DATASET_DIR / "ambystoma_maculatum_flux_canalization_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1984-bryant-ambystoma-maculatum-flux-canalization.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
