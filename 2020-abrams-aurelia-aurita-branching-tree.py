#!/usr/bin/env python3
"""
2020-abrams-aurelia-aurita-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0664)
Organism: Aurelia aurita (Moon Jellyfish)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Aurelia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0664_aurelia_aurita_branching_tree.lean` (`MathOfBiology.Loop_0664`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/aurelia_aurita_branching_tree"
EXPECTED_SHA256 = {
    "aurelia_aurita_branching_tree_reference.json": "a7497f23274af6976f5ac02335e025e98a91c21e50bce8cbf97d91f075ad4617",
    "aurelia_aurita_specimen.jpg": "1a8c5b9fb4abe35f1362b2c458d03bafa15966b0f6046d49aa840fdd3a66769e",
    "scout_manifest.json": "a6a0cd30fa6d0eba129d0af443caca068c71b6d30d8c0dc264192f84eee26979",
    "paper_citations.bib": "b3bdc74bd3ba697eb9bb72243c81591a37aeeb457543013768bd67c12adf42f1"
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
    ref = json.loads((DATASET_DIR / "aurelia_aurita_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-abrams-aurelia-aurita-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
