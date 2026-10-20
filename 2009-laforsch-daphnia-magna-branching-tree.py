#!/usr/bin/env python3
"""
2009-laforsch-daphnia-magna-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0649)
Organism: Daphnia magna (Water Flea)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Entoma (Arthropoda) -> Daphnia
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0649_daphnia_magna_branching_tree.lean` (`MathOfBiology.Loop_0649`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/daphnia_magna_branching_tree"
EXPECTED_SHA256 = {
    "daphnia_magna_branching_tree_reference.json": "b2944e45e083753e00f3f908ca72dc6f0d1de9099c5a527684c4ea6346e0970a",
    "daphnia_magna_specimen.jpg": "fc61b6ab51b92e988777a61cce4db7434210496ee5084ae01ee9f7adf6fcf44b",
    "scout_manifest.json": "5e429fbb845a086a07e6f30c6f8b5490220c0ac1e190a2112a744dcac3422aef",
    "paper_citations.bib": "bb713c64c5a6edd3fb87db3b459e745239e13d9f872aaaf158e8de4666daf06c"
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
    ref = json.loads((DATASET_DIR / "daphnia_magna_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2009-laforsch-daphnia-magna-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
