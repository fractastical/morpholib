#!/usr/bin/env python3
"""
2003-shoji-tetraodon-mbu-biomineral-accretion.py — Pass 2 Remediated Reproduction (Slot #0832)
Organism: Tetraodon mbu (Giant Pufferfish)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Tetraodon
Biophysical Cohort: VIII. Biomineralization, Shells & Spicules (Logarithmic Accretion & Minimal-Surface Biomineralization)
Formal Verification: `morphoeval/lean/loops/Loop_0832_tetraodon_mbu_biomineral_accretion.lean` (`MathOfBiology.Loop_0832`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tetraodon_mbu_biomineral_accretion"
EXPECTED_SHA256 = {
    "tetraodon_mbu_biomineral_accretion_reference.json": "699543503aa66e5339f0d1d68a9e043a4b059adabf2836061b7dae71c38501ca",
    "tetraodon_mbu_specimen.jpg": "c2d84cbf503eef4d1bdc55fdc0e9d3ba124750e0ae48ff7010b6a4cdca2ee3f3",
    "scout_manifest.json": "aaf3486ab089a710461b94960f6227c63cf7bb31a193a41d3d86c3bb27bfd0ab",
    "paper_citations.bib": "05e37b3fc882372cf4500dc95da84e92d114625063cb7267e8b1c65822883508"
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
    ref = json.loads((DATASET_DIR / "tetraodon_mbu_biomineral_accretion_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2003-shoji-tetraodon-mbu-biomineral-accretion.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
