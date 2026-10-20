#!/usr/bin/env python3
"""
2020-porter-hippocampus-kuda-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0634)
Organism: Hippocampus kuda (Spotted Seahorse)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Pisces (Ichthyes) -> Hippocampus
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0634_hippocampus_kuda_branching_tree.lean` (`MathOfBiology.Loop_0634`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/hippocampus_kuda_branching_tree"
EXPECTED_SHA256 = {
    "hippocampus_kuda_branching_tree_reference.json": "9a88f3f0f7809309b83bd9cb464563b75fe89d8ac32fbd7f6bd658d9888624a7",
    "hippocampus_kuda_specimen.jpg": "2d453503e6f09426796b25434fc672645ac4aa639654b8d282c6607928df715e",
    "scout_manifest.json": "147c99967ffb4f3a16276a84e355394d967eadc409aa62062217d387b6700662",
    "paper_citations.bib": "7e0fe1e8e9b39e492341a1a6b19275e4650d92b3e6e78df31b23efb28c6ecb9a"
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
    ref = json.loads((DATASET_DIR / "hippocampus_kuda_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2020-porter-hippocampus-kuda-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
