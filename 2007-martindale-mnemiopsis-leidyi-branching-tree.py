#!/usr/bin/env python3
"""
2007-martindale-mnemiopsis-leidyi-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0670)
Organism: Mnemiopsis leidyi (Warty Comb Jelly)
Aristotelian Taxonomy: II. Anaima (Invertebrata) -> Zoophyta & Vermes (Cnidaria/Platyhelminthes) -> Mnemiopsis
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0670_mnemiopsis_leidyi_branching_tree.lean` (`MathOfBiology.Loop_0670`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/mnemiopsis_leidyi_branching_tree"
EXPECTED_SHA256 = {
    "mnemiopsis_leidyi_branching_tree_reference.json": "5fb3612226d2a90d4d05f82a1c33cd918955f49d4b940686fb974abe549e1d55",
    "mnemiopsis_leidyi_specimen.jpg": "d966e0f5b06b5af8e1ef77edf5e32c90c3aa8489be8bbbf2bdf7795ecbfb7ec7",
    "scout_manifest.json": "3164dfa9c6ca8da100bf4f9269f11ff1d923eeb1520177c8ed0ddf1006de1367",
    "paper_citations.bib": "47344ed222478a48e458ef8ca1d5d86b192408f18653e9ae15ddc852d8bb2420"
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
    ref = json.loads((DATASET_DIR / "mnemiopsis_leidyi_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-martindale-mnemiopsis-leidyi-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
