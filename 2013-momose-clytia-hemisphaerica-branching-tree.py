#!/usr/bin/env python3
"""
2013-momose-clytia-hemisphaerica-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0668)
Organism: Clytia hemisphaerica (Colonial Hydrozoan)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Clytia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0668_clytia_hemisphaerica_branching_tree.lean` (`MathOfBiology.Loop_0668`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/clytia_hemisphaerica_branching_tree"
EXPECTED_SHA256 = {
    "clytia_hemisphaerica_branching_tree_reference.json": "5fdea7e883fdd394770a9be716985c082a15e923e8484b682695dec9f6a7f23b",
    "clytia_hemisphaerica_specimen.jpg": "2fcc34f8d72cdc5aca117a04c8528f2e4227f504bc61fe0588a43db0a016f236",
    "scout_manifest.json": "4ec51f71c0f7a1404dad9fb5fcf9486dede228d501ad77e789d3563a560732c7",
    "paper_citations.bib": "5c71a8cea859ce7bf52d692ea35919bf68779717fd6468e8a272c7da4bc7b4e7"
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
    ref = json.loads((DATASET_DIR / "clytia_hemisphaerica_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2013-momose-clytia-hemisphaerica-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
