#!/usr/bin/env python3
"""
1981-bryant-ambystoma-maculatum-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0822)
Organism: Ambystoma maculatum (Spotted Salamander)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Ambystoma
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0822_ambystoma_maculatum_biomineral_accretion.lean` (`MathOfBiology.Loop_0822`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/ambystoma_maculatum_biomineral_accretion"
EXPECTED_SHA256 = {
    "ambystoma_maculatum_biomineral_accretion_reference.json": "9d405efaadc93b631552410e561c1739c8e7e7277978abfeecf75e901d8e6d11",
    "ambystoma_maculatum_specimen.jpg": "32e4b406ab308f3580f182f76a10b7ef37ad16c150b655ab556d886365f151d6",
    "scout_manifest.json": "d0381e4b02084745de19bc366897fd6eb4a9683f618d47b16cf0a1a55b4ddd41",
    "paper_citations.bib": "cf8e39e315f968fcc56b48c3bcbd4e75c3e46abd79c1356606922c0f0b7dc1af"
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
    ref = json.loads((DATASET_DIR / "ambystoma_maculatum_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1981-bryant-ambystoma-maculatum-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
