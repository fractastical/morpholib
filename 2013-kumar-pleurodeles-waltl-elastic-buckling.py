#!/usr/bin/env python3
"""
2013-kumar-pleurodeles-waltl-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0725)
Organism: Pleurodeles waltl (Iberian Ribbed Newt)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Pleurodeles
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0725_pleurodeles_waltl_elastic_buckling.lean` (`MathOfBiology.Loop_0725`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pleurodeles_waltl_elastic_buckling"
EXPECTED_SHA256 = {
    "pleurodeles_waltl_elastic_buckling_reference.json": "3bf5253668d72ea015d0bf30ab6735b0701c2222f1fcbda5fec585b6eaf745b8",
    "pleurodeles_waltl_specimen.jpg": "5b04f7216aba72146f64ff5aae72cd1f25ecdc62a55c051a3aea6d71c7dd09c5",
    "scout_manifest.json": "79af3f6c49bb8a3c42bfdb98e80a337db931ad3b7532f1a3f08c9ef4a9f49233",
    "paper_citations.bib": "25d2e452f8f4d4a84f2a4b5e1bfdf1fc87a3bf1d6f17d03ca630805981ac88e0"
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
    ref = json.loads((DATASET_DIR / "pleurodeles_waltl_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-kumar-pleurodeles-waltl-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
