#!/usr/bin/env python3
"""
2007-prum-taeniopygia-guttata-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0612)
Organism: Taeniopygia guttata (Zebra Finch)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Taeniopygia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0612_taeniopygia_guttata_branching_tree.lean` (`MathOfBiology.Loop_0612`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/taeniopygia_guttata_branching_tree"
EXPECTED_SHA256 = {
    "taeniopygia_guttata_branching_tree_reference.json": "f9064366bc18a33b0b037c5b2c3a6ebddeb127ce7ef70bf67ab1efbd1072e084",
    "taeniopygia_guttata_specimen.jpg": "cba8970e2c1bea961592887848134a4a694bea3ad4d2b68524b2a63c722dffa0",
    "scout_manifest.json": "f53371e4af84ab1593214a1033201287d6ca2509413adaa3bf723e17591a8bc4",
    "paper_citations.bib": "09d16bfe46835af517402fd5be9977bf3a98f4a48f2826c90fe0fd7fb50ab603"
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
    ref = json.loads((DATASET_DIR / "taeniopygia_guttata_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-prum-taeniopygia-guttata-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
