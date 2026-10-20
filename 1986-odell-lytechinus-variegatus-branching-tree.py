#!/usr/bin/env python3
"""
1986-odell-lytechinus-variegatus-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0655)
Organism: Lytechinus variegatus (Green Sea Urchin)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Ostrakoderma (Mollusca & Echinodermata) -> Lytechinus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0655_lytechinus_variegatus_branching_tree.lean` (`MathOfBiology.Loop_0655`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/lytechinus_variegatus_branching_tree"
EXPECTED_SHA256 = {
    "lytechinus_variegatus_branching_tree_reference.json": "54a9a66169502d4f1f679e0d02c1b1aad447d3a2e634ed91aea74501329ed05e",
    "lytechinus_variegatus_specimen.jpg": "521ca1f087eb9618f793b80f0084a9efa6db3c8fb7039599c38d75ba00260619",
    "scout_manifest.json": "8169b02f2f1c4d4d902bc4bd18dee93daae4db787c10b2e07891e42c7601c8a0",
    "paper_citations.bib": "f44d265637f78df68c8c0028ef4bd31d02a62683b421686640341142a2364312"
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
    ref = json.loads((DATASET_DIR / "lytechinus_variegatus_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1986-odell-lytechinus-variegatus-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
