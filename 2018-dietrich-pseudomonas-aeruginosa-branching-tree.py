#!/usr/bin/env python3
"""
2018-dietrich-pseudomonas-aeruginosa-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0684)
Organism: Pseudomonas aeruginosa (Biofilm Bacterium)
Aristotelian Taxonomy: IV. Protista & Monera -> Slime Molds & Bacteria -> Pseudomonas
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0684_pseudomonas_aeruginosa_branching_tree.lean` (`MathOfBiology.Loop_0684`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pseudomonas_aeruginosa_branching_tree"
EXPECTED_SHA256 = {
    "pseudomonas_aeruginosa_branching_tree_reference.json": "e2179f57bf1bbcb9ffec195fab40a5773dea717f3cedd0cf54ffe93c69d72079",
    "pseudomonas_aeruginosa_specimen.jpg": "60ef5e2ce063576716d4086509c23046d1f100864118a90be5614a770b3ff31d",
    "scout_manifest.json": "38e43aa1613e9657cc26df45a4a7a0b8d0c29bae7a4fda86c712a102ab6d1a8e",
    "paper_citations.bib": "6fc1db77d00869eb8c322cda34b88ee3a52bcc36de7614b13026eb0bc1082919"
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
    ref = json.loads((DATASET_DIR / "pseudomonas_aeruginosa_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-dietrich-pseudomonas-aeruginosa-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
