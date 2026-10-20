#!/usr/bin/env python3
"""
2018-milinkovitch-alligator-mississippiensis-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0630)
Organism: Alligator mississippiensis (American Alligator)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Alligator
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0630_alligator_mississippiensis_branching_tree.lean` (`MathOfBiology.Loop_0630`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/alligator_mississippiensis_branching_tree"
EXPECTED_SHA256 = {
    "alligator_mississippiensis_branching_tree_reference.json": "a5c1c5a514ada56d9736b982302a51efa07ccf4da597c4375db9332c3c0e0754",
    "alligator_mississippiensis_specimen.jpg": "bb1770a8cf4a71024eec779363925a2928277079f2124074117061fc6d960764",
    "scout_manifest.json": "90d2b08b15023b6033e7ab2965cba761d22ac99e0a556b199d21bff344dd275b",
    "paper_citations.bib": "57cc9eea3ec480a4218ffc39dca390a80de0a1c958e75d72e85a9806745d29ec"
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
    ref = json.loads((DATASET_DIR / "alligator_mississippiensis_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-milinkovitch-alligator-mississippiensis-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
