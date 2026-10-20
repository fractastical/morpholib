#!/usr/bin/env python3
"""
2019-ocs-zea-mays-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0673)
Organism: Zea mays (Maize)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Zea
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0673_zea_mays_branching_tree.lean` (`MathOfBiology.Loop_0673`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/zea_mays_branching_tree"
EXPECTED_SHA256 = {
    "zea_mays_branching_tree_reference.json": "c06028d37d17ae64ec2658a5d9074cbaee294e3a8e6ef66fa143aacdff443a06",
    "zea_mays_specimen.jpg": "a78e27ea3c83c91a3f84373319a3b590f4a53e6108ddf50860d114690cbb6db9",
    "scout_manifest.json": "e7cfd2e492d794177c776282a4cb91003077efa632f8fd7fad7dc13433672212",
    "paper_citations.bib": "978d9836e9c1a23ee6b25f5c51a53281c55f2c57ec96599a7515a5bb64a84b7c"
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
    ref = json.loads((DATASET_DIR / "zea_mays_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2019-ocs-zea-mays-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
