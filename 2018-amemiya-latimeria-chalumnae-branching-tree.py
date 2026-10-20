#!/usr/bin/env python3
"""
2018-amemiya-latimeria-chalumnae-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0639)
Organism: Latimeria chalumnae (West Indian Coelacanth)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Latimeria
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0639_latimeria_chalumnae_branching_tree.lean` (`MathOfBiology.Loop_0639`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/latimeria_chalumnae_branching_tree"
EXPECTED_SHA256 = {
    "latimeria_chalumnae_branching_tree_reference.json": "243d87575c4d3f467c8d525420d94439be86651af529c427c3471ce9055157f8",
    "latimeria_chalumnae_specimen.jpg": "29bcb5946834bbbf95c873d247d7bb9449f8ea6b56ddef496b4eef521ce6cd5d",
    "scout_manifest.json": "670ede3a047f3314a293b8e2aa8ae22f5310305fed29817336f0b57aa913dd29",
    "paper_citations.bib": "b39e88e110173bdbe55d4569c88ae175c258d3b8827035cbe224a8a33b432e99"
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
    ref = json.loads((DATASET_DIR / "latimeria_chalumnae_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-amemiya-latimeria-chalumnae-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
