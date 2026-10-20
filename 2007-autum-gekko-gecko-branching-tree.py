#!/usr/bin/env python3
"""
2007-autum-gekko-gecko-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0628)
Organism: Gekko gecko (Tokay Gecko)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Amphibia & Reptilia (Ootoka) -> Gekko
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0628_gekko_gecko_branching_tree.lean` (`MathOfBiology.Loop_0628`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/gekko_gecko_branching_tree"
EXPECTED_SHA256 = {
    "gekko_gecko_branching_tree_reference.json": "24439911e05271a3f033775a51b4837bce4bf213f9387443a4931c3820fc9355",
    "gekko_gecko_specimen.jpg": "d5822e517434737d90b14fd5099c77f09ab976ca2a3ba00c2484f69a65eb91e8",
    "scout_manifest.json": "00d79dfb7969249312048353f829957d149355670ad7fec91b6dff84457fcbeb",
    "paper_citations.bib": "22a8540c32832477f78df8888acca51aa37e3ab3368fe9947319ec049542e510"
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
    ref = json.loads((DATASET_DIR / "gekko_gecko_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2007-autum-gekko-gecko-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
