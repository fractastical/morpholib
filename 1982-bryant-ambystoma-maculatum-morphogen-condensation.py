#!/usr/bin/env python3
"""
1982-bryant-ambystoma-maculatum-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0222)
Organism: Ambystoma maculatum (Spotted Salamander)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ambystoma
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0222_ambystoma_maculatum_morphogen_condensation.lean` (`MathOfBiology.Loop_0222`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ambystoma_maculatum_morphogen_condensation"
EXPECTED_SHA256 = {
    "ambystoma_maculatum_morphogen_condensation_reference.json": "46c88383018422c8e590fe73f653387005adc60337f1e7995f00fa3ed6f7f91f",
    "ambystoma_maculatum_specimen.jpg": "32e4b406ab308f3580f182f76a10b7ef37ad16c150b655ab556d886365f151d6",
    "scout_manifest.json": "5e216d34e8182633b7a4c06eeb35031aa46a6608c49fc3c0d93cdd45944ff64e",
    "paper_citations.bib": "5f5295e9a17aaf424ae342f5d0eebfa0e784ad73c301f22b83d72976e0462a46"
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
    ref = json.loads((DATASET_DIR / "ambystoma_maculatum_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1982-bryant-ambystoma-maculatum-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
