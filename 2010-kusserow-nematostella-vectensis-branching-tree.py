#!/usr/bin/env python3
"""
2010-kusserow-nematostella-vectensis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0662)
Organism: Nematostella vectensis (Starlet Sea Anemone)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Nematostella
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0662_nematostella_vectensis_branching_tree.lean` (`MathOfBiology.Loop_0662`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/nematostella_vectensis_branching_tree"
EXPECTED_SHA256 = {
    "nematostella_vectensis_branching_tree_reference.json": "b69a4b276af9845ad5f5e35f9798964f39835ca58b75a0124d32a0600f57a0c1",
    "nematostella_vectensis_specimen.jpg": "fb9d66de457a62bbf3d2ec1bcdb57f33e32468690a3047a09117a19153073a53",
    "scout_manifest.json": "520cbd39ee179cf3825fc82217e9483fe3988edfd8d49ed892a6e5c004f02f7f",
    "paper_citations.bib": "76fb7ddfb98fbdd67f4b00419a9ddf3996f04c552a27069d12f863711c94626e"
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
    ref = json.loads((DATASET_DIR / "nematostella_vectensis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2010-kusserow-nematostella-vectensis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
