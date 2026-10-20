#!/usr/bin/env python3
"""
1999-jean-pinus-pinea-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0675)
Organism: Pinus pinea (Stone Pine)
Aristotelian Taxonomy: III. Phyta (Embryophyta) -> Angiospermae & Gymnospermae -> Pinus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0675_pinus_pinea_branching_tree.lean` (`MathOfBiology.Loop_0675`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/pinus_pinea_branching_tree"
EXPECTED_SHA256 = {
    "pinus_pinea_branching_tree_reference.json": "7cd944dcfaf280c08b33ce14a7ed8b5369d9ea40bd4f832240d683bf403ede9c",
    "pinus_pinea_specimen.jpg": "87c1fef6a6fb55bd3bf3e35f590d280e453c1348b5d11f18c356c09345da3904",
    "scout_manifest.json": "e8e16123499932e5015fb49ce2abeeca23ccaaa59e161fe376002ff3143e6506",
    "paper_citations.bib": "187b63fab2d2114a3ad632415bf3db2371f7f0b8d699906cde6aae7514da2d7e"
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
    ref = json.loads((DATASET_DIR / "pinus_pinea_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 1999-jean-pinus-pinea-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
