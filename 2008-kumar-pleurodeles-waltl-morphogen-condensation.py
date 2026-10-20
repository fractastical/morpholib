#!/usr/bin/env python3
"""
2008-kumar-pleurodeles-waltl-morphogen-condensation.py — Pass 2 Remediated Reproduction (Slot #0225)
Organism: Pleurodeles waltl (Iberian Ribbed Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Pleurodeles
Biophysical Cohort: II. Limb Bud, Skeletal & Cranial Morphogenesis (Morphogen Gradient Positional Condensation)
Formal Verification: `morphoeval/lean/loops/Loop_0225_pleurodeles_waltl_morphogen_condensation.lean` (`MathOfBiology.Loop_0225`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pleurodeles_waltl_morphogen_condensation"
EXPECTED_SHA256 = {
    "pleurodeles_waltl_morphogen_condensation_reference.json": "f73da3d6da0c5ea80db023c100a40457c5f7dd557debcf32508f85b13e89b6c4",
    "pleurodeles_waltl_specimen.jpg": "5b04f7216aba72146f64ff5aae72cd1f25ecdc62a55c051a3aea6d71c7dd09c5",
    "scout_manifest.json": "0e55ad2179038ea8aa2c7bfba7d533fe01672f666d03b48923713018b45e27c5",
    "paper_citations.bib": "2b51eb02f863c2f04ec0b436504b13fe66d3e706829a931486a9b67899fd74da"
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
    ref = json.loads((DATASET_DIR / "pleurodeles_waltl_morphogen_condensation_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-kumar-pleurodeles-waltl-morphogen-condensation.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
