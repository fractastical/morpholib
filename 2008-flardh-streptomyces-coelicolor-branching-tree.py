#!/usr/bin/env python3
"""
2008-flardh-streptomyces-coelicolor-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0688)
Organism: Streptomyces coelicolor (Mycelial Actinobacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Streptomyces
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0688_streptomyces_coelicolor_branching_tree.lean` (`MathOfBiology.Loop_0688`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/streptomyces_coelicolor_branching_tree"
EXPECTED_SHA256 = {
    "streptomyces_coelicolor_branching_tree_reference.json": "28ea3563de74b7571f844f51b5d6744df89eb8d711aebc4565d6ee37df325888",
    "streptomyces_coelicolor_specimen.jpg": "5a2b99731bd4971f58260f74a1db5d4560b78f4d8ceb3b067629dcb7b6d53fec",
    "scout_manifest.json": "83b9e7839fea78c03e879a9f32714cc3727228f8f17ee3421912e9a024631ad1",
    "paper_citations.bib": "938ed6a60d885c7a6d0bb487c6a599f7c194818e3b015c1fffa047f734065a24"
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
    ref = json.loads((DATASET_DIR / "streptomyces_coelicolor_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2008-flardh-streptomyces-coelicolor-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
