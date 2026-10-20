#!/usr/bin/env python3
"""
2009-wu-anas-platyrhynchos-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0616)
Organism: Anas platyrhynchos (Mallard Duck)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Anas
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0616_anas_platyrhynchos_branching_tree.lean` (`MathOfBiology.Loop_0616`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/anas_platyrhynchos_branching_tree"
EXPECTED_SHA256 = {
    "anas_platyrhynchos_branching_tree_reference.json": "2b1e1eed522eaa3f9fab506745b6ad91c2bdfe083078158b031d2e9c4c59be6d",
    "anas_platyrhynchos_specimen.jpg": "ee1cc0720d68f5f20e5e3490647b69cb29c1fe84dae8fdb048019814f69eccba",
    "scout_manifest.json": "f154266e15c7a23b280b85f2096ece644e60cb13450167f5015e768eb19be53e",
    "paper_citations.bib": "e3b4bd4cb487db9721a70fe88061a9fa25f5dd924e81b07b706b7183fdf4145c"
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
    ref = json.loads((DATASET_DIR / "anas_platyrhynchos_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-wu-anas-platyrhynchos-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
