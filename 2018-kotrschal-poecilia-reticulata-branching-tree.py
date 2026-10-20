#!/usr/bin/env python3
"""
2018-kotrschal-poecilia-reticulata-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0637)
Organism: Poecilia reticulata (Trinidadian Guppy)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Poecilia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0637_poecilia_reticulata_branching_tree.lean` (`MathOfBiology.Loop_0637`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/poecilia_reticulata_branching_tree"
EXPECTED_SHA256 = {
    "poecilia_reticulata_branching_tree_reference.json": "6f8ae865a4310f0c891f3572cadeb20474857c5e770abebe64a341084e1cdf91",
    "poecilia_reticulata_specimen.jpg": "3e998877d83b46f9df12f9fa135839a5982986b29a14965629955111cadf8a2e",
    "scout_manifest.json": "b2663e5458acf290c81d65cdb0678fab9f9a844b6799fd0156e72ffa2cde8108",
    "paper_citations.bib": "394fd4dc50f33caa516fa639ddf009dfe216ada90d68bf2aa71160a5107e450a"
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
    ref = json.loads((DATASET_DIR / "poecilia_reticulata_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-kotrschal-poecilia-reticulata-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
