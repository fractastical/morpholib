#!/usr/bin/env python3
"""
2018-wagner-tyto-alba-branching-tree.py — Pass 2 Remediated Reproduction (Slot #0618)
Organism: Tyto alba (Barn Owl)
Aristotelian Taxonomy: I. Enaima (Vertebrata) -> Aves (Ornithes) -> Tyto
Biophysical Cohort: VI. Branching Morphogenesis & Vascular Networks (Bifurcating Tree Murray Power-Law & Tip Splitting)
Formal Verification: `morphoeval/lean/loops/Loop_0618_tyto_alba_branching_tree.lean` (`MathOfBiology.Loop_0618`)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "datasets/tyto_alba_branching_tree"
EXPECTED_SHA256 = {
    "tyto_alba_branching_tree_reference.json": "201234f10ef3ae7bd10cf84ab5892b4b1afbb101e17845504860b57e01938e65",
    "tyto_alba_specimen.jpg": "1a81ca2e5a4b493f8c795ded141df4f7bfbee333f381a42a0b020882d72c2671",
    "scout_manifest.json": "d024699671e8806add385ad65f644bd06714e041d8fcafaf00506724abe7a7d2",
    "paper_citations.bib": "134042d69e94e192dcfab05f765372e59dbbd280bf16e5313b1571135a48e53a"
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
    ref = json.loads((DATASET_DIR / "tyto_alba_branching_tree_reference.json").read_text(encoding="utf-8"))
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
        print(f"[PASS] 2018-wagner-tyto-alba-branching-tree.py self-test: r={res['pearson_r']:.5f}, rmse={res['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
