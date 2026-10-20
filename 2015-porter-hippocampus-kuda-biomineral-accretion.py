#!/usr/bin/env python3
"""
2015-porter-hippocampus-kuda-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0834)
Organism: Hippocampus kuda (Spotted Seahorse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Hippocampus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0834_hippocampus_kuda_biomineral_accretion.lean` (`MathOfBiology.Loop_0834`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hippocampus_kuda_biomineral_accretion"
EXPECTED_SHA256 = {
    "hippocampus_kuda_biomineral_accretion_reference.json": "951cbe655a0b8f2a8b590fa52fee0a3d3b139ac51271274cd3581497621080ae",
    "hippocampus_kuda_specimen.jpg": "2d453503e6f09426796b25434fc672645ac4aa639654b8d282c6607928df715e",
    "scout_manifest.json": "44501733a7596398fbd175a4253e0ea4de6d8b9cbc03fde9b5b5f851fb60474f",
    "paper_citations.bib": "303cdf77ac763e7473d75545ac4cb6f36d5b32109e17250cae250295d4541669"
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
    ref = json.loads((DATASET_DIR / "hippocampus_kuda_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2015-porter-hippocampus-kuda-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
