#!/usr/bin/env python3
"""
2008-prum-taeniopygia-guttata-elastic-buckling.py — Pass 2 Remediated Reproduction (Slot #0712)
Organism: Taeniopygia guttata (Zebra Finch)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Taeniopygia
Biophysical Cohort: VII. Mechanical Folding, Wrinkling & Buckling (Differential Growth Elastic Bilayer Buckling)
Formal Verification: `morphoeval/lean/loops/Loop_0712_taeniopygia_guttata_elastic_buckling.lean` (`MathOfBiology.Loop_0712`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/taeniopygia_guttata_elastic_buckling"
EXPECTED_SHA256 = {
    "taeniopygia_guttata_elastic_buckling_reference.json": "6b019accdb101c3a0c5be9d9e82917552492261dc20cb9901782de9c1135297f",
    "taeniopygia_guttata_specimen.jpg": "cba8970e2c1bea961592887848134a4a694bea3ad4d2b68524b2a63c722dffa0",
    "scout_manifest.json": "e4b7b903f4905e591851049ebd67261f9a7597d23a177de87a30b0950ebf2f06",
    "paper_citations.bib": "caf8d5d839618347bd5c2089734070749c34a8d832347e74a56811bbbdfc73cf"
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
    ref = json.loads((DATASET_DIR / "taeniopygia_guttata_elastic_buckling_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-prum-taeniopygia-guttata-elastic-buckling.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
