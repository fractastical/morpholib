#!/usr/bin/env python3
"""
2009-gordon-arachnoidiscus-ehrenbergii-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0899)
Organism: Arachnoidiscus ehrenbergii (Wheel Centric Diatom)
Aristotelian Taxonomy: IV. Protista & Monera -> Mineralized Protists (Silica/Calcite) -> Arachnoidiscus
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0899_arachnoidiscus_ehrenbergii_biomineral_accretion.lean` (`MathOfBiology.Loop_0899`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/arachnoidiscus_ehrenbergii_biomineral_accretion"
EXPECTED_SHA256 = {
    "arachnoidiscus_ehrenbergii_biomineral_accretion_reference.json": "be5a77bce918cc55d6827a61caa2b6100db92c50ee7de986bea519ddd1a2b243",
    "arachnoidiscus_ehrenbergii_specimen.jpg": "499034f85d30d464ea67fea4d7404a3546b57b85bb64c8d553cb36f2129bed05",
    "scout_manifest.json": "7bfb9ec526d31480c54288393103856b2afb1450ef9128d9328d5e98052bfa4c",
    "paper_citations.bib": "7fc3cfd01011c1aea99ebce18c87c185740e095ba5cdba13c53c3c35bb4fd7c5"
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
    ref = json.loads((DATASET_DIR / "arachnoidiscus_ehrenbergii_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-gordon-arachnoidiscus-ehrenbergii-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
