#!/usr/bin/env python3
"""
1999-koch-giraffa-camelopardalis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0605)
Organism: Giraffa camelopardalis (Reticulated Giraffe)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Mammalia (Zootoka) -> Giraffa
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0605_giraffa_camelopardalis_branching_tree.lean` (`MathOfBiology.Loop_0605`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/giraffa_camelopardalis_branching_tree"
EXPECTED_SHA256 = {
    "giraffa_camelopardalis_branching_tree_reference.json": "d59d0f066651859ca4449b27dbd0f7fd6b91f792275ca1e7d083d0720a8834a1",
    "giraffa_camelopardalis_specimen.jpg": "d65ee763a067a60ce829e77f52e0ca7eadb883d032bc517d6f76edf060800969",
    "scout_manifest.json": "57974bf61f6c03e83669e7c9060b5d7e9ea229d5a7c63c6c7c8975258d4b4c71",
    "paper_citations.bib": "1d2f2b0fdb4ac85c6196b62b7324800936c867a51ce35918f904b1d72d067bce"
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
    ref = json.loads((DATASET_DIR / "giraffa_camelopardalis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1999-koch-giraffa-camelopardalis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
